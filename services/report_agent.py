# services/report_agent.py
import os
import tiktoken
import openai
import pandas as pd
from IPython.display import display, Markdown
from dotenv import load_dotenv
from typing import List, Tuple, Union, Any, Optional
from pydantic import BaseModel, Field
import nest_asyncio
import json

from operator import itemgetter

from llama_index.core import Settings, set_global_tokenizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.tools.types import BaseTool
from llama_index.core.tools import ToolSelection
from llama_index.core.response_synthesizers import TreeSummarize, CompactAndRefine
from llama_index.core.workflow import Event
from llama_index.core.prompts import Prompt

# Enable async support
nest_asyncio.apply()

load_dotenv()

# Validate OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Check your environment variables.")

# Set global tokenizer for LlamaIndex
set_global_tokenizer(tiktoken.encoding_for_model("gpt-4").encode)

# Define data models for report blocks
class TextBlock(BaseModel):
    text: str = Field(..., description="The text for this block.")

class TableBlock(BaseModel):
    caption: str = Field(..., description="Caption of the table.")
    col_names: List[str] = Field(..., description="Names of the columns.")
    rows: List[Tuple] = Field(..., description="List of rows with data entry tuples.")

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=self.col_names)
        return df

report_gen_system_prompt = """\
You are a report generation assistant tasked with producing a well-formatted report given parsed context.
You will be given context from one or more reports that take the form of parsed text + tables
You are responsible for producing a report with interleaving text and tables - in the format of interleaving text and "table" blocks.

Make sure the report is detailed with a lot of textual explanations especially if tables are given.

You MUST output your response as a tool call in order to adhere to the required output format. Do NOT give back normal text.

Here is an example of a toy valid tool call - note the text and table block:
{
    "blocks": [
        {
            "text": "A report on cities"
        },
        {
            "caption": "Comparison of CityA vs. CityB",
            "col_names": [
              "",
              "Population",
              "Country",
            ],
            "rows": [
              [
                "CityA",
                "1,000,000",
                "USA"
              ],
              [
                "CityB",
                "2,000,000",
                "Mexico"
              ]
            ]
        }
    ]
}
"""

class ReportOutput(BaseModel):
    blocks: List[Union[TextBlock, TableBlock]] = Field(..., description="A list of blocks: text, table.")

    def render(self) -> None:
        for block in self.blocks:
            if isinstance(block, TextBlock):
                display(Markdown(block.text))
            elif isinstance(block, TableBlock):
                display(block.to_df())

# Updated system prompt for report generation
report_gen_system_prompt = """\
You are a report generation assistant tasked with producing a well-formatted report given parsed context.

You will be given context from one or more documents that include text and possibly tables.
Your job:
- Produce a final report in **valid JSON format** that conforms to the ReportOutput schema below.
- The final JSON must contain a "blocks" key, which is an array of objects.
- Each object in "blocks" can be one of the following:
  - {"text": "..."}
  - {"caption": "...", "col_names": [...], "rows": [...]}

Return only the JSON object with no extra text or formatting. Ensure it's valid JSON.
"""

# Initialize embeddings and LLM
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4o")
Settings.embed_model = embed_model
Settings.llm = llm

report_gen_llm = OpenAI(model="gpt-4o", max_tokens=2048, system_prompt=report_gen_system_prompt)
# Convert to a structured LLM, but we will still parse JSON just in case
report_gen_sllm = report_gen_llm.as_structured_llm(output_cls=ReportOutput)

# Define workflow events
class InputEvent(Event):
    input: List[ChatMessage]

class ChunkRetrievalEvent(Event):
    tool_call: ToolSelection

class DocRetrievalEvent(Event):
    tool_call: ToolSelection

class ReportGenerationEvent(Event):
    pass

class ReportGenerationAgent(Workflow):
    def __init__(
        self,
        chunk_retriever_tool: BaseTool,
        doc_retriever_tool: BaseTool,
        llm: Optional[FunctionCallingLLM] = None,
        report_gen_sllm: Optional[StructuredLLM] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.chunk_retriever_tool = chunk_retriever_tool
        self.doc_retriever_tool = doc_retriever_tool
        self.llm = llm or OpenAI()
        self.summarizer = CompactAndRefine(llm=self.llm)
        assert self.llm.metadata.is_function_calling_model
        self.report_gen_sllm = report_gen_sllm or self.llm.as_structured_llm(
            ReportOutput, system_prompt=report_gen_system_prompt
        )
        self.report_gen_summarizer = TreeSummarize(llm=self.report_gen_sllm)

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)

    @step(pass_context=True)
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        self.sources = []
        ctx.data["stored_chunks"] = []
        ctx.data["query"] = ev.input
        user_msg = ChatMessage(role="user", content=ev.input)
        self.memory.put(user_msg)
        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step(pass_context=True)
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> Union[ChunkRetrievalEvent, DocRetrievalEvent, ReportGenerationEvent, StopEvent]:
        chat_history = ev.input
        response = await self.llm.achat_with_tools(
            [self.chunk_retriever_tool, self.doc_retriever_tool],
            chat_history=chat_history,
        )
        self.memory.put(response.message)
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if not tool_calls:
            return ReportGenerationEvent()
        for tool_call in tool_calls:
            if tool_call.tool_name == self.chunk_retriever_tool.metadata.name:
                return ChunkRetrievalEvent(tool_call=tool_call)
            elif tool_call.tool_name == self.doc_retriever_tool.metadata.name:
                return DocRetrievalEvent(tool_call=tool_call)
        return ReportGenerationEvent()

    @step(pass_context=True)
    async def handle_retrieval(
        self, ctx: Context, ev: Union[ChunkRetrievalEvent, DocRetrievalEvent]
    ) -> InputEvent:
        """Handle retrieval."""
        query = ev.tool_call.tool_kwargs["query"]
        retrieved_chunks = (
            self.chunk_retriever_tool(query).raw_output
            if isinstance(ev, ChunkRetrievalEvent)
            else self.doc_retriever_tool(query).raw_output
        )
        ctx.data["stored_chunks"].extend(retrieved_chunks)

        response = self.summarizer.synthesize(query, nodes=retrieved_chunks)
        self.memory.put(
            ChatMessage(
                role="tool",
                content=str(response),
                additional_kwargs={
                    "tool_call_id": ev.tool_call.tool_id,
                    "name": ev.tool_call.tool_name,
                },
            )
        )

        return InputEvent(input=self.memory.get())

    @step(pass_context=True)
    async def generate_report(self, ctx: Context, ev: ReportGenerationEvent) -> StopEvent:
        """Generate report."""
        response = self.report_gen_summarizer.synthesize(
            ctx.data["query"], nodes=ctx.data["stored_chunks"]
        )
        print(f"Report generation response: {response}")
        return StopEvent(result={"response": response})

# The main function to run the agent
async def run_agent(input_query: str, index_name: str):
    print("Running agent...", input_query)
    index = LlamaCloudIndex(
        name=index_name,
        project_name="Default",
        api_key=os.getenv("LLAMA_CLOUD_API_KEY")
    )
    doc_retriever = index.as_retriever(retrieval_mode="files_via_content", files_top_k=1)
    chunk_retriever = index.as_retriever(retrieval_mode="chunks", rerank_top_n=5)

    def chunk_retriever_fn(query: str) -> List[NodeWithScore]:
        return chunk_retriever.retrieve(query)

    def doc_retriever_fn(query: str) -> List[NodeWithScore]:
        return doc_retriever.retrieve(query)

    chunk_retriever_tool = FunctionTool.from_defaults(fn=chunk_retriever_fn)
    doc_retriever_tool = FunctionTool.from_defaults(fn=doc_retriever_fn)

    agent = ReportGenerationAgent(
        chunk_retriever_tool,
        doc_retriever_tool,
        llm=llm,
        report_gen_sllm=report_gen_sllm,
        verbose=True,
        timeout=120.0,
    )

    ret = await agent.run(input=input_query)
    # ret is {"response": ReportOutput}
    result: ReportOutput = ret["response"]  # This should now be a ReportOutput object
    return result

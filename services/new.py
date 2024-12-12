#new.py
import os
import tiktoken
import openai
import pandas as pd
from IPython.display import display, Markdown
from dotenv import load_dotenv
from typing import List, Tuple, Union, Any, Optional
from operator import itemgetter
from pydantic import BaseModel, Field

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
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize, CompactAndRefine
from llama_index.core.workflow import Event

# Load environment variables
load_dotenv()

# Validate OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Check your environment variables.")

# Set global tokenizer
set_global_tokenizer(tiktoken.encoding_for_model("gpt-4o").encode)

# Enable async in non-notebook environments
import nest_asyncio
nest_asyncio.apply()

# Setup embedding and LLM models
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4o")
Settings.embed_model = embed_model
Settings.llm = llm

# Define function tools
class TextBlock(BaseModel):
    """Text block."""
    text: str = Field(..., description="The text for this block.")

class TableBlock(BaseModel):
    """Table block."""
    caption: str = Field(..., description="Caption of the table.")
    col_names: List[str] = Field(..., description="Names of the columns.")
    rows: List[Tuple] = Field(
        ..., description="List of rows with data entry tuples."
    )

    def to_df(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame(self.rows, columns=self.col_names)
        df.style.set_caption(self.caption)
        return df

class ReportOutput(BaseModel):
    """Data model for a report."""
    blocks: List[Union[TextBlock, TableBlock]] = Field(
        ..., description="A list of text and table blocks."
    )

    def render(self) -> None:
        """Render as formatted text within a notebook."""
        for block in self.blocks:
            if isinstance(block, TextBlock):
                display(Markdown(block.text))
            else:
                display(block.to_df())

# Define report generation prompt
report_gen_system_prompt = """\
You are a report generation assistant tasked with producing a well-formatted report given parsed context.
You will be given context from one or more reports that take the form of parsed text + tables.
You are responsible for producing a report with interleaving text and tables.
Make sure the report is detailed with a lot of textual explanations, especially if tables are given.
"""

# Initialize structured LLM
report_gen_llm = OpenAI(model="gpt-4o", max_tokens=2048, system_prompt=report_gen_system_prompt)
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

# Define ReportGenerationAgent class
class ReportGenerationAgent(Workflow):
    """Report generation agent."""

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
        self.sources = []

    @step(pass_context=True)
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        """Prepare chat history."""
        self.sources = []
        ctx.data["stored_chunks"] = []
        ctx.data["query"] = ev.input

        user_msg = ChatMessage(role="user", content=ev.input)
        self.memory.put(user_msg)

        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step(pass_context=True)
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> Union[ChunkRetrievalEvent, DocRetrievalEvent, ReportGenerationEvent, StopEvent]:
        """Handle LLM input."""
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
            return ReportGenerationEvent(input=ev.input)

        for tool_call in tool_calls:
            if tool_call.tool_name == self.chunk_retriever_tool.metadata.name:
                return ChunkRetrievalEvent(tool_call=tool_call)
            elif tool_call.tool_name == self.doc_retriever_tool.metadata.name:
                return DocRetrievalEvent(tool_call=tool_call)
            else:
                return StopEvent(result={"response": "Invalid tool."})

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
        return StopEvent(result={"response": response})

# Define async function to run the agent
async def run_agent(input: str, index_name: str):
    # Instantiate the LlamaCloudIndex with the provided name
    index = LlamaCloudIndex(
        name=index_name,
        project_name="Default",
        api_key=os.getenv("LLAMA_CLOUD_API_KEY")
    )

    # Define retrievers dynamically based on the index
    doc_retriever = index.as_retriever(retrieval_mode="files_via_content", files_top_k=1)
    chunk_retriever = index.as_retriever(retrieval_mode="chunks", rerank_top_n=5)

    # Update function tools to use the dynamic retrievers
    def chunk_retriever_fn(query: str) -> List[NodeWithScore]:
        """Retrieve relevant document chunks."""
        return chunk_retriever.retrieve(query)

    def doc_retriever_fn(query: str) -> float:
        """Retrieve entire documents."""
        return doc_retriever.retrieve(query)

    chunk_retriever_tool = FunctionTool.from_defaults(fn=chunk_retriever_fn)
    doc_retriever_tool = FunctionTool.from_defaults(fn=doc_retriever_fn)

    # Instantiate the agent with dynamic tools
    agent = ReportGenerationAgent(
        chunk_retriever_tool,
        doc_retriever_tool,
        llm=llm,
        report_gen_sllm=report_gen_sllm,
        verbose=True,
        timeout=120.0,
    )

    # Run the agent
    ret = await agent.run(input=input)
    return ret

# Main function
async def main():
    index_name = "project_e3aab84f-d507-412b-8f95-ae79589e8893_index"  # Replace with the desired index name
    ret = await run_agent("Summarize LOFTQ", index_name)
    print(ret)

# Run the main function
import asyncio
asyncio.run(main())

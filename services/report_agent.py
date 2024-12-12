# services/report_agent.py
import os
import tiktoken
import openai
import pandas as pd
from IPython.display import display, Markdown
from dotenv import load_dotenv
from typing import List, Tuple, Union, Any
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
from llama_index.prompts import PromptTemplate

# Enable async support
nest_asyncio.apply()

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

set_global_tokenizer(tiktoken.encoding_for_model("gpt-4").encode)

# Define data models for report blocks
class TextBlock(BaseModel):
    text: str = Field(..., description="The text for this block.")

class TableBlock(BaseModel):
    caption: str = Field(..., description="Caption of the table.")
    col_names: List[str] = Field(..., description="Names of the columns.")
    rows: List[Tuple] = Field(..., description="List of rows with data entry tuples.")
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.col_names)

class ChartBlock(BaseModel):
    type: str = Field(..., description="Type of the chart (bar, line, pie).")
    title: str = Field(..., description="Title of the chart.")
    data: dict = Field(..., description="Data for the chart, e.g. {\"x\":[], \"y\":[]}")

class ImageBlock(BaseModel):
    image_description: str = Field(..., description="A description of the image.")

class ReportOutput(BaseModel):
    blocks: List[Union[TextBlock, TableBlock, ChartBlock, ImageBlock]] = Field(
        ..., description="A list of blocks: text, table, chart, image."
    )

    def render(self) -> None:
        for block in self.blocks:
            if isinstance(block, TextBlock):
                display(Markdown(block.text))
            elif isinstance(block, TableBlock):
                display(block.to_df())
            elif isinstance(block, ChartBlock):
                # For demonstration, just display the data
                display(Markdown(f"**Chart Title:** {block.title}\n**Chart Type:** {block.type}\nData: {block.data}"))
            elif isinstance(block, ImageBlock):
                display(Markdown(f"**Image Description:** {block.image_description}"))

# Updated system prompt for report generation
report_gen_system_prompt = """\
You are a report generation assistant tasked with producing a well-formatted report given parsed context.

You will be given context from one or more documents that include text and possibly tables, charts, and images.
Your job:
- Produce a final report in **valid JSON format** that conforms to the ReportOutput schema below.
- The final JSON must contain a "blocks" key, which is an array of objects.
- Each object in "blocks" can be one of the following:
  - {"text": "..."} for textual explanation
  - {"caption": "...", "col_names": [...], "rows": [...]} for a table
  - {"type": "bar|line|pie", "title": "...", "data": {"x":[], "y":[]}} for a chart
  - {"image_description": "..."} for an image block

Return only the JSON object, no extra text or formatting. Ensure it's valid JSON.
"""

# Initialize embeddings and LLM
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4")
Settings.embed_model = embed_model
Settings.llm = llm

report_gen_llm = OpenAI(model="gpt-4", max_tokens=2048, system_prompt=report_gen_system_prompt)
report_gen_sllm = report_gen_llm.as_structured_llm(output_cls=ReportOutput)

# Functions to convert ReportOutput to different formats
def generate_pdf(report: ReportOutput) -> str:
    # Placeholder logic - implement as needed
    # For example, use fpdf to create a PDF from the blocks
    pdf_path = "reports/report.pdf"
    # ... code to generate PDF ...
    return pdf_path

def generate_docx(report: ReportOutput) -> str:
    # Placeholder logic - implement as needed using python-docx
    docx_path = "reports/report.docx"
    # ... code to generate DOCX ...
    return docx_path

def generate_html(report: ReportOutput) -> str:
    # Placeholder logic - implement HTML generation
    html_path = "reports/report.html"
    # ... code to generate HTML ...
    return html_path

def generate_markdown(report: ReportOutput) -> str:
    # Placeholder logic - implement markdown generation
    md_path = "reports/report.md"
    # ... code to generate Markdown ...
    return md_path

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
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> Union[ChunkRetrievalEvent, DocRetrievalEvent, ReportGenerationEvent]:
        chat_history = ev.input
        response = await self.llm.achat_with_tools(
            [self.chunk_retriever_tool, self.doc_retriever_tool],
            chat_history=chat_history,
        )
        self.memory.put(response.message)
        tool_calls = self.llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
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
        query = ctx.data["query"]
        if isinstance(ev, ChunkRetrievalEvent):
            retrieved_chunks = ev.tool_call.tool_fn(query)
        else:
            retrieved_chunks = ev.tool_call.tool_fn(query)

        print("Retrieved Chunks:", retrieved_chunks)
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
        query = ctx.data["query"]
        nodes = ctx.data["stored_chunks"]

        # Extract text from nodes
        node_text = ""
        for ns in nodes:
            node_content = ns.node.get_content()
            if node_content:
                node_text += node_content + "\n\n"

        print("Node Text:", node_text)

        if not node_text.strip():
            print("No node_text found, returning a fallback.")
            return StopEvent(result={"response": "No retrieved data available."})

        user_prompt = f"""
        You have the following context:
        {node_text}

        The user asks: {query}

        Please produce the final report in valid JSON format following the ReportOutput schema previously described.
        Return only the JSON object, no extra text or formatting. Ensure it's valid JSON.
        """

        prompt_template = PromptTemplate(template=user_prompt.strip())
        raw_text = await self.report_gen_sllm.apredict(prompt=prompt_template)
        raw_text = raw_text.strip()

        print("Raw LLM Output:", raw_text)

        if not raw_text:
            raise ValueError("LLM returned empty response, cannot parse JSON.")

        try:
            response_dict = json.loads(raw_text)
            report_output = ReportOutput.parse_obj(response_dict)
        except Exception as e:
            print("Failed to parse JSON output:", e)
            print("Raw Output for Debugging:", raw_text)
            raise ValueError(f"Failed to parse LLM output into ReportOutput: {e}")
        
        return StopEvent(result={"response": report_output})


async def run_agent(input_query: str, index_name: str, output_format: str = "JSON") -> Any:
    print("Running agent with query:", input_query)
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
    result = ret["response"]  # This should be a ReportOutput

    # Convert report to desired format if requested
    if isinstance(result, ReportOutput):
        if output_format.upper() == "PDF":
            return generate_pdf(result)
        elif output_format.upper() == "DOCX":
            return generate_docx(result)
        elif output_format.upper() == "HTML":
            return generate_html(result)
        elif output_format.upper() == "MARKDOWN":
            return generate_markdown(result)
        else:
            # Default: return JSON (the ReportOutput object itself)
            return result
    else:
        # Try to parse if not already
        return ReportOutput.parse_obj(result)


# Example usage (run in an async environment):
# import asyncio
# result = asyncio.run(run_agent("Summarize LOFTQ", "project_your_index_name", output_format="PDF"))
# print("Final Result:", result)

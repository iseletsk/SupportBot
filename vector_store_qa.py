from typing import List, Dict

from langchain.agents.agent_toolkits.vectorstore.prompt import ROUTER_PREFIX
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import Field

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import AgentExecutor, ZeroShotAgent

from langchain.agents.agent_toolkits import VectorStoreRouterToolkit
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.llms import BaseLLM
from langchain.tools import BaseTool
from langchain.tools.vectorstore.tool import VectorStoreQAWithSourcesTool, BaseVectorStoreTool
from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreInfo,
)
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings



class VectorStoreWithSourcesRouterToolkit(VectorStoreRouterToolkit):
    """Toolkit for routing between vectorstores."""


    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools: List[BaseTool] = []
        for vectorstore_info in self.vectorstores:
            description = VectorStoreQAWithSourcesTool.get_description(
                vectorstore_info.name, vectorstore_info.description
            )
            qa_tool = VectorStoreQAWithSourcesTool(
                name=vectorstore_info.name,
                description=description,
                vectorstore=vectorstore_info.vectorstore,
                llm=self.llm,

            )
            tools.append(qa_tool)
        return tools


def init_original_vectorstore_agent(docs: List[Dict], verbose:bool) -> AgentExecutor:
    infos = [VectorStoreInfo(name=doc["prefix"], description=doc["description"], vectorstore=doc["vectordb"]) for doc in docs]
    llm = OpenAI(temperature=0, verbose=verbose)
    toolkit = VectorStoreRouterToolkit(vectorstores=infos, llm=llm, verbose=verbose)
    return create_vectorstore_router_agent(llm=llm, toolkit=toolkit, verbose=verbose, return_intermediate_steps=True)


def init_vectorstore_agent(docs: List[Dict], verbose: bool) -> AgentExecutor:
    infos = [VectorStoreInfo(name=doc["prefix"], description=doc["description"], vectorstore=doc["vectordb"]) for doc in docs]
    llm = OpenAI(temperature=0, verbose=verbose)
    toolkit = VectorStoreWithSourcesRouterToolkit(vectorstores=infos, llm=llm, verbose=verbose)
    return create_vectorstore_router_agent(llm=llm, toolkit=toolkit, verbose=verbose, return_intermediate_steps=True)



class MyQATool(BaseVectorStoreTool, BaseTool):
    sources = []

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name}. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            "Input should be a fully formed question."
        )
        return template.format(name=name, description=description)

    def reset(self):
        self.sources = []

    def _get_chain(self) -> Chain:
        chain = RetrievalQA.from_chain_type(self.llm, retriever=HydeRetriever(vectorstore=self.vectorstore),
                                            return_source_documents=True)
        return chain

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = self._get_chain()
        result = chain({"query": query})
        self.sources.extend(result["source_documents"])
        return result["result"]

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""

        chain = self._get_chain()
        result = await chain._acall({"query": query})
        self.sources.extend(result["source_documents"])
        return result["result"]

class MyRoutingToolkit(BaseToolkit):
    """Toolkit for routing between vectorstores."""

    vectorstores: List[VectorStoreInfo] = Field(exclude=True)
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools: List[BaseTool] = []
        for vectorstore_info in self.vectorstores:
            description = MyQATool.get_description(vectorstore_info.name, vectorstore_info.description)
            qa_tool = MyQATool(name=vectorstore_info.name, description=description, vectorstore=vectorstore_info.vectorstore)
            tools.append(qa_tool)
        return tools


def init_retrieval_vectorstore_agent(docs: List[Dict], verbose: bool) -> (AgentExecutor, List[MyQATool]):
    infos = [VectorStoreInfo(name=doc["prefix"], description=doc["description"], vectorstore=doc["vectordb"]) for doc in docs]
    llm = OpenAI(temperature=0, verbose=verbose)
    chat_llm = OpenAI(model='gpt-3.5-turbo', temperature=0, verbose=verbose)

    toolkit = MyRoutingToolkit(vectorstores=infos, llm=chat_llm, verbose=verbose)
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(tools, prefix=ROUTER_PREFIX)
    llm_chain = LLMChain(llm=llm, prompt=prompt, callback_manager=None,)
    tool_names = [tool.name for tool in tools]
    RETURN_INTERMEDIATE_STEPS = True
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, verbose=verbose,
                          return_intermediate_steps=RETURN_INTERMEDIATE_STEPS)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose,
                                              return_intermediate_steps=RETURN_INTERMEDIATE_STEPS), tools


class HydeRetriever(VectorStoreRetriever):
    hyde: HypotheticalDocumentEmbedder = None
    embeddings: Embeddings = None
    llm: BaseLLM = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings()
        if self.llm is None:
            self.llm = OpenAI()

        prompt_template = "Write a documentation answering the question.\nQuestion: {QUESTION}\nDocumentation:"
        prompt = PromptTemplate(template=prompt_template, input_variables=["QUESTION"])
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        self.hyde = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=self.embeddings)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        vector = self.hyde.embed_query(query)
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search_by_vector(vector, **self.search_kwargs)
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search_by_vector(vector, **self.search_kwargs)
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        vector = self.hyde.embed_query(query) ## Todo implement async

        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search_by_vector(
                vector, **self.search_kwargs
            )
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search_by_vector(
                vector, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
from langchain.agents import AgentExecutor


def init_custom_agent(docs, verbose: bool) -> AgentExecutor:
    from langchain.agents import initialize_agent, Tool
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    from langchain.agents import AgentType

    llm = OpenAI(temperature=0)
    tools = []
    for doc in docs:
        rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc["vectordb"].as_retriever())
        #rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc["vectordb"].as_retriever())
        tools.append(Tool(name=doc["prefix"], description=doc["description"], func=rqa.run))

    '''
            agent: A string that specified the agent type to use. Valid options are:
            `zero-shot-react-description`
            `react-docstore`
            `self-ask-with-search`
            `conversational-react-description`
            `chat-zero-shot-react-description`,
            `chat-conversational-react-description`,
           If None and agent_path is also None, will default to
            `zero-shot-react-description`.
    '''
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=verbose, return_intermediate_steps=True)
    agent.return_intermediate_steps = True

    return agent


def init_custom_chat_agent(docs, verbose: bool) -> AgentExecutor:
    from langchain.agents import initialize_agent, Tool
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    from langchain.agents import AgentType
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory

    llm_chat = ChatOpenAI(temperature=0, verbose=verbose)
    llm = OpenAI(temperature=0, verbose=verbose)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    tools = []
    for doc in docs:
        rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc["vectordb"].as_retriever(verbose=verbose), verbose=verbose)

        #rqa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=doc["vectordb"].as_retriever())
        tools.append(Tool(name=doc["prefix"], description=doc["description"], func=rqa.run))

    '''
            agent: A string that specified the agent type to use. Valid options are:
            `zero-shot-react-description`
            `react-docstore`
            `self-ask-with-search`
            `conversational-react-description`
            `chat-zero-shot-react-description`,
            `chat-conversational-react-description`,
           If None and agent_path is also None, will default to
            `zero-shot-react-description`.
    '''
    agent = initialize_agent(tools=tools, llm=llm_chat, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                             verbose=verbose, memory=memory, return_intermediate_steps=True)
    return agent

#agent_executor = init_custom_agent(docs)
#agent_executor = init_custom_chat_agent(docs)

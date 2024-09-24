import traceback
from typing import Any, Dict, List, Optional
from itertools import groupby
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain, ConversationalRetrievalChain, ConversationChain, create_qa_with_sources_chain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers.retry import RetryOutputParser
from langchain.schema import OutputParserException, PromptValue
from langchain_openai import OpenAI


# Additional imports for database drivers and system instructions
from driver import redisdb, weaviatedb
from uuid import uuid4
from langchain_core.tracers.context import collect_runs

# from llm.conversationchain import ConversationChain, ConversationalRetrievalChain
# from embedding.systeminstruct import SystemInstruct
# mapping title with gglink
    
class LangChainBot:
    __history_configure = False
    __knowledge_configure = False
    #

    def __init__(
            self,
            debug=False,
            init_params: Dict[str, Any] = {},
    ):
        # self.retry_parser, self.prompt_value = self.create_retry_parser_for_final_answer()
        self.debug = debug
        # print(f"LangChainBot:\tInitialization")
    #

    @classmethod
    def bare_init(
            cls,
            # system_instruction : Dict[str, Optional[Dict[str, Any]]] = {
            # 	"instruct_embedding" : SystemInstruct,
            # },
            history_store: Dict[str, Optional[Dict[str, Any]]] = {
                "history_driver": redisdb.RedisDB,
            },
            knowledge_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "knowledge_driver": weaviatedb.WeaviateDB,
            },
            condense_question_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "chain_core": LLMChain,
                # "llm_core" : ChatOpenAI,
                "llm_core": ChatOllama,
                "llm_core_params": {
                    "temperature": 0,
                    "model": "phi",
                }
            },
            combine_docs_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "stuff_chain_core": StuffDocumentsChain,
                "chain_core": LLMChain,
                # "llm_core" : OpenAI,
                "llm_core": Ollama,
                "llm_core_params": {
                    "temperature": 0,
                    "model": "phi",
                }
            },
            memory_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "memory_core": ConversationBufferMemory,
            },
            stack_chain: Dict[str, Optional[Dict[str, Any]]] = {
                "chain_core": ConversationalRetrievalChain,
                "runnable_chain": RunnableWithMessageHistory,
            },
            embedding_configure: Dict[str, Optional[Dict[str, Any]]] = {
                'embedding': OpenAIEmbeddings()
            }
    ):
        try:
            bot = cls()
            #
            # if "instruct_embedding" in system_instruction:
            # 	bot.system_instruction(**system_instruction)
            #
            if "knowledge_driver" in knowledge_configure:
                bot.knowledge_configure(**knowledge_configure)
            # Use default or specific hsitory store
            if "history_driver" in history_store:
                bot.history_store(**history_store)
            # Or connect to the history store of api app (available)
            elif "history_store" in history_store:
                bot.set_history_store(**history_store)
            # Chose llm embedding
            bot.embedding_configure(**embedding_configure)
            # Chain to RAG: combine user question and session history to query revelant documents from vectorstore.
            bot.condense_question_configure(**condense_question_configure)
            # Memory: Only using ConversationBufferMemory as default to handle recent conversation, no chat history setup here
            bot.memory_configure(**memory_configure)
            # Chain to interpreter: combine user question and returned revelant documents to produce answer.
            bot.combine_docs_configure(**combine_docs_configure)
            # Chatbot RAG chain: stacking with chat history, RAG chain and interpreter chain
            bot.stack_chain(**stack_chain)
            return bot
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            return None
    #

    def embedding_configure(
            self,
            embedding
    ):
        self.__llm_embedding = embedding
        # print(f"LangChainBot:\t embedding_configure: {embedding}")
        return True

    # def system_instruction(
    # 	self,
    # 	instruct_embedding: Any = SystemInstruct,
    # 	instruct_embedding_params: Optional[Dict[str, Any]] = {},
    # ) -> bool:
    # 	try:
    # 		instruct = instruct_embedding(**instruct_embedding_params)
    # 		if not instruct or not instruct.get_sample():
    # 			raise Exception(f"LangChainBot:\tSystem instruction can not configured.")
    # 		self.__instruct = instruct
    # 		self.__system_instruction = True
    # 		print(f"LangChainBot:\tsystem instruction: {instruct_embedding}")
    # 		return True
    # 	except Exception as e:
    # 		print(str(e))
    # 		self.__instruct = None
    # 		self.__system_instruction = False
        # return False
    #
    def knowledge_configure(
            self,
            knowledge_driver: Any = weaviatedb.WeaviateDB,
            knowledge_driver_params: Optional[Dict[str, Any]] = {},
    ) -> bool:
        try:
            driver = knowledge_driver(**knowledge_driver_params)
            if not driver.is_connected():
                raise Exception(
                    f"LangChainBot:\tKnowledge database can not connected.")
            self.__retriever = driver
            self.__knowledge_configure = True
            # print(
            #     f"LangChainBot:\tknowledge base retriever: {knowledge_driver}")
            return True
        except Exception as e:
            # print(f'Cannot connect self.__retriever with weaviate {str(e)}')
            self.__retriever = None
            self.__knowledge_configure = False
            return False

    def history_store(
            self,
            history_driver: Any = redisdb.RedisDB,
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            driver = history_driver()
            if not driver.is_connected():
                raise Exception(
                    f"LangChainBot:\tHistory database is not connected.")
            self.__history = driver
            self.__history_configure = True
            # print(f"LangChainBot:\tchat history retriever: {history_driver}")
            return True
        except Exception as e:
            print(str(e))
            self.__history = None
            self.__history_configure = False
            return False
    #

    def has_history_store(self,):
        return self.__history_configure == True
    #

    def get_history_store(self,):
        if not self.__history_configure:
            return None
        return self.__history
    #

    def set_history_store(
            self,
            history_store,
            overwrite: bool = True,
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            if not self.__history_configure or overwrite:
                if not history_store.is_connected():
                    raise Exception(
                        f"LangChainBot:\tHistory database is not connected.")
                self.__history = history_store
                self.__history_configure = True
            # print(f"LangChainBot:\tchat history store: using FastAPIApp history store.")
            return True
        except Exception as e:
            print(str(e))
            self.__history = None
            self.__history_configure = False
            return False
    #

    def condense_question_configure(
            self,
            chain_core,
            llm_core,
            prompt_core_template,
            llm_core_params: Dict[str, Any] = {},
            # prompt_core_template: str = """Chat History: {chat_history}\n\nUser question: {question}""",
    ) -> bool:
        try:
            chain = LLMChain(
                llm=llm_core(**llm_core_params),
                prompt=PromptTemplate.from_template(prompt_core_template)
            )
            if not chain:
                raise Exception(f"LangChainBot:\tCan not init {llm_core}")
            self.__condense_question_chain = chain
            # print(
            #     f"LangChainBot:\tCondense question chain configure: {chain_core} --- {llm_core}")
            return True
        except Exception as e:
            print(str(e))
            self.__condense_question_chain = None
            return False
    #

    def combine_docs_configure(
            self,
            stuff_chain_core: StuffDocumentsChain,
            chain_core: LLMChain,
            prompt_core_template,
            llm_core,
            llm_core_params: Dict[str, Any] = {},
    ) -> bool:
        if self.debug:
            llm_chain = chain_core(
                llm=llm_core(**llm_core_params),
                prompt=PromptTemplate.from_template(prompt_core_template),
                verbose=True,
            )
            if not llm_chain:
                raise Exception(f"LangChainBot:\tCan not init {chain_core}")
            #
            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                template="""
				\n Recipe information: \n {page_content}\n {metadata}"""
            )
            chain = stuff_chain_core(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name="context",
                verbose=True,
                # call_back =
            )
            if not chain:
                raise Exception(
                    f"LangChainBot:\tCan not init {stuff_chain_core}")
            self.__combine_docs_chain = chain
            # print(
            #     f"LangChainBot:\tCombine documents chain configure: {chain_core} --- {llm_core}")
            return True
        else:
            try:
                
                llm_chain = chain_core(
                    llm=llm_core(**llm_core_params),
                    prompt=PromptTemplate.from_template(prompt_core_template),
                    verbose=False,
                )
                if not llm_chain:
                    raise Exception(
                        f"LangChainBot:\tCan not init {chain_core}")
                #
                document_prompt = PromptTemplate(
                    input_variables=["page_content","metadata"],
                    template="""
                    \n Recipe information: \n {page_content}\n {metadata}"""
                )
                chain = stuff_chain_core(
                    llm_chain=llm_chain,
                    document_prompt=document_prompt,
                    document_variable_name="context",
                    verbose=False,
                )
                if not chain:
                    raise Exception(
                        f"LangChainBot:\tCan not init {stuff_chain_core}")
                self.__combine_docs_chain = chain
                # print(
                #     f"LangChainBot:\tCombine documents chain configure: {chain_core} --- {llm_core}")
                return True

            except Exception as e:
                print(str(e))
                self.__combine_docs_chain = None
                return False
        #

    def memory_configure(
            self,
            memory_core: ConversationBufferMemory,
            memory_core_params: Dict[str, Any] = {},
    ) -> bool:
        try:
            # memory = memory_core(**memory_core_params)
            # Try ConversationBufferMemory() with default initial
            memory = memory_core(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True,
                # **memory_core_params,
            )
            # print(f"LangChainBot:\tMemory configure: {memory_core}")
            self.__memory = memory
            self.__memory_configure = True
            return True
        except Exception as e:
            print(str(e))
            self.__memory = None
            self.__memory_configure = False
            return False
    #

    def create_retry_parser_for_final_answer(self):
        final_answer = ResponseSchema(
            name="final_answer",
            description="return 'no value' if the anwser is 'None' else return the answer",
        )
        output_parser = StructuredOutputParser.from_response_schemas(
            [final_answer]
        )
        retry_parser = RetryOutputParser.from_llm(
            llm=OpenAI(temperature=0),
            parser=output_parser,
            max_retries=3
        )
        prompt = PromptTemplate(
            template="Reply the query\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()},
        )
        prompt_value = prompt.format_prompt(
            query='Find the last answer, provide the evidence and reasoning of this answer.')
        return retry_parser, prompt_value
    def chain_constructor(self,tenant_name,index_db,text_key ):
            new_memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer")            
            return ConversationalRetrievalChain(
                retriever=self.__retriever.get_langchain_vectorstore(as_retriever=True,
                                                                     tentant_name=tenant_name,
                                                                     index_name=index_db,
                                                                     text_key=text_key,
                                                                     embedding=self.__llm_embedding,
                                                                     k = 4
                                                                     ),
                memory = new_memory, 
                question_generator=self.__condense_question_chain,
                # Chat
                combine_docs_chain=self.__combine_docs_chain,)
    def stack_chain(
            self,
            chain_core: ConversationalRetrievalChain,
            runnable_chain: Optional[RunnableWithMessageHistory] = None,
            tenant_name: str = "Admin",
            index_db: str = "None",
            text_key: str = 'text',
    ) -> bool:
        
        # self.evaluation_chain = self.chain_constructor(tenant_name,index_db,text_key)
        try:
            ##
            qa = ConversationalRetrievalChain(
                # Retriever
                retriever=self.__retriever.get_langchain_vectorstore(as_retriever=True,
                                                                     tentant_name=tenant_name,
                                                                     index_name=index_db,
                                                                     text_key=text_key,
                                                                     embedding=self.__llm_embedding,
                                                                     k = 5),
                # memory = self.__memory, #use redis memory so I do not use ConversationBufferMemory
                question_generator=self.__condense_question_chain,
                # Chat
                combine_docs_chain=self.__combine_docs_chain,  # custome with stuff_chain_core
                return_source_documents=True,  # modified
                verbose=True,
                # verbose = False,
            )
            # self.evaluation_chain = chain_constructor()
            # Deal with chat history using session_id
            if runnable_chain and self.has_history_store():
                # print('--------WE USE RunnableWithMessageHistory')
                chain = RunnableWithMessageHistory(
                    qa,
                    lambda session_id: self.__history.get_langchain_chat_message_history(
                        session_id=session_id),
                    input_messages_key="question",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
                self.__runnable_chain = True
            # Without chat history, buffer memory only
            else:
                # print('--------WE buffer memory only')
                chain = qa
                self.__runnable_chain = False
            ##
            if not chain:
                raise Exception(f"Can not init {chain_core}")
            # chain stacking
            self.__chain = chain
            # For evaluation

            # self.evaluation_chain = chain
            self.__stack_chain = True
            # print(f"LangChainBot:\tchain: {chain_core}")
            return True
        except Exception as e:
            print(str(e))
            self.__chain = None
            self.__stack_chain = False
            return False
    # API service

    async def ask(
            self,
            session_id: str,
            question: str,

    ):
        try:
            msg = ""
            if self.__runnable_chain:
                # print('LANGCHAINBOT|	WE,RE USING SESSION ID')
                output = await self.__chain.ainvoke(
                    {
                        "question": question,  
                    },
                    config={"configurable": {"session_id": session_id}},
                    # include_run_info= True, 
                )
            else:
                output = await self.__chain(
                    {
                        "question": question,
                    },
                )
            # print(f'------The answer response: {output}')
            # print(f'------The run_id: {run_id}')
            answer = f'{output["answer"]}'
            # result = self.retry_parser.parse_with_prompt(str(output), self.prompt_value)
            answer = {
                'chatbot_answer': f'{output["answer"]}',
                
            }
            return answer, msg
        except Exception as e:
            print('Chatbot error: ',str(e))
            traceback.print_exc()
            # return {"bot" : "this is a sample response for debugger"}, ""
            return "_", e
    
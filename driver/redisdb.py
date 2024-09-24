import os
from typing import Optional, Any, List, Dict, Union
from uuid import uuid4

import redis

from langchain_community.chat_message_histories import RedisChatMessageHistory

REDIS_HOST = "REDIS_HOST"
REDIS_PORT = "REDIS_PORT"
REDIS_PWD = "REDIS_PWD"

# DB_ID : type int or str from [0 .. 15]
REDIS_DB = "REDIS_DB"


class RedisDB:
    __host: str = None
    __port: str = None
    __pwd: str = None
    __connected: bool = False
    __client: Dict[int, redis.Redis] = {}
    #

    def __init__(
            self,
    ) -> None:
        try:
            self.__getEnvironmentVariables()
            # session_id : List[str], self.__client[0]
            if not self.connect(db=0, with_pwd=True):
                self.connect(db=0, with_pwd=False)
            # if self.__client:
            # 	## user_id : {session_id, }
            # 	self.connect(db = 1, with_pwd = self.__with_pwd)
            # 	## (user_id, session_id) : true/false
            # 	self.connect(db = 2, with_pwd = self.__with_pwd)
        except Exception as e:
            self.disconnect()
            print(str(e))
    #

    def __getEnvironmentVariables(self,) -> None:
        self.__host = os.environ.get(REDIS_HOST) or "localhost"
        self.__port = os.environ.get(REDIS_PORT) or "6379"
        # client required connection config
        self.__pwd = os.environ.get(REDIS_PWD) or None

    def connect(self, db: int = 0, with_pwd: bool = True) -> bool:
        # print("----------redis self.__host:", self.__host)
        try:
            db = 0 if (db < 0 or db > 15) else db
            if with_pwd and self.__pwd != None:
                self.__client[db] = redis.Redis(
                    host=self.__host,
                    port=int(self.__port),
                    password=self.__pwd,
                    db=db
                )
                self.__with_pwd = True
            else:
                self.__client[db] = redis.Redis(
                    host=self.__host,
                    port=self.__port,
                    db=db
                )
                self.__with_pwd = False
            if not self.__client[db].ping():
                return False
            self.__connected = True
            return True
        except Exception as e:
            print(f'Cannot connect with redis: {str(e)}')
            return False
    #

    def disconnect(self,):
        if self.__connected:
            for client in self.__client:
                client.close()
        self.__client = None
        self.__connected = False
    #

    def check_connection(self,) -> bool:
        try:
            if not self.__client[0].ping():
                return False
            return True
        except Exception as e:
            print("Connection failed:", str(e))
            return False
    #

    def is_connected(self) -> bool:
        return self.__connected
    #

    def has_key_value(
            self,
            key: Any,
            db: int = 0,
    ) -> bool:
        if not self.__connected:
            return False
        value = self.__client[db].get(key)
        if not value:
            return False
        return True
    #

    def get_client(self, db: int = 0) -> redis.Redis:
        if not self.__client or db in self.__client:
            self.connect(db=db, with_pwd=self.__with_pwd)
        return self.__client[db]
    #

    def get_url(self, db: int = 0) -> str:
        if not self.__connected or db not in self.__client:
            return ""
        url = "redis://"
        url += ":" + self.__pwd + "@" if self.__pwd else ""
        url += self.__host + ":" + self.__port
        url += "/" + str(db)
        # print(url)
        return url
    #

    def get_langchain_chat_message_history(
            self,
            session_id: str = "None",
            key_prefix: str = "message_store:",
            ttl: int = 600
    ) -> Any:
        if session_id == "None":
            session_id = uuid4().hex
        try:
            chathistory = RedisChatMessageHistory(
                url=self.get_url(db=0),
                session_id=session_id,
                key_prefix=key_prefix,
                ttl=ttl,
            )
            return chathistory
        except Exception as e:
            print(str(e))
            return None

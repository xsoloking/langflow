from datetime import timedelta
from typing import List

from langchain_community.vectorstores import ElasticVectorSearch

from langflow.base.vectorstores.model import LCVectorStoreComponent
from langflow.helpers.data import docs_to_data
from langflow.io import HandleInput, IntInput, StrInput, SecretStrInput, DataInput, MultilineInput
from langflow.schema import Data


class ElasticsearchVectorStoreComponent(LCVectorStoreComponent):
    display_name = "Elasticsearch"
    description = "Elasticsearch Vector Store with search capabilities"
    documentation = "https://python.langchain.com/v0.1/docs/integrations/document_loaders/elasticsearch/"
    name = "Elasticsearch"
    icon = "Redis"

    inputs = [
        StrInput(
            name="elasticsearch_url", display_name="Elasticsearch Cluster url", required=True
        ),
        StrInput(name="elasticsearch_username", display_name="Elasticsearch username", required=True),
        SecretStrInput(name="elasticsearch_password", display_name="Elasticsearch password", required=True),
        StrInput(name="index_name", display_name="Index Name", required=True),
        MultilineInput(name="search_query", display_name="Search Query"),
        DataInput(
            name="ingest_data",
            display_name="Ingest Data",
            is_list=True,
        ),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
            advanced=True,
        ),
    ]

    def build_vector_store(self) -> ElasticVectorSearch:
        return self._build_elasticsearch()

    def _build_elasticsearch(self) -> ElasticVectorSearch:
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)

        if documents:
            elasticsearch_vs = ElasticVectorSearch.from_documents(
                elasticsearch_url=self.elasticsearch_url,
                http_auth=(self.elasticsearch_username, self.elasticsearch_password),
                verify_certs=False,
                documents=documents,
                embedding=self.embedding,
                index_name=self.index_name,
            )
        else:
            elasticsearch_vs = ElasticVectorSearch(
                elasticsearch_url=self.elasticsearch_url,
                http_auth=(self.elasticsearch_username, self.elasticsearch_password),
                verify_certs=False,
                embedding=self.embedding,
                index_name=self.index_name,
            )

        return elasticsearch_vs

    def search_documents(self) -> List[Data]:
        vector_store = self._build_elasticsearch()

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            docs = vector_store.similarity_search(
                query=self.search_query,
                k=self.number_of_results,
            )

            data = docs_to_data(docs)
            self.status = data
            return data
        else:
            return []

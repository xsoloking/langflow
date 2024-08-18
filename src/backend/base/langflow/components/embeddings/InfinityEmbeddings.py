from langchain_community.embeddings import InfinityEmbeddings

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import Embeddings
from langflow.io import MessageTextInput, Output


class InfinityEmbeddingsComponent(LCModelComponent):
    display_name: str = "Infinity Embeddings"
    description: str = "Generate embeddings using Infinity models."
    documentation = "https://python.langchain.com/docs/integrations/text_embedding/infinity"
    icon = "Redis"
    name = "InfinityEmbeddings"

    inputs = [
        MessageTextInput(
            name="model",
            display_name="Infinity Model",
            value="lier007/xiaobu-embedding-v2",
        ),
        MessageTextInput(
            name="base_url",
            display_name="Infinity Base URL",
            value="http://infinity.192.168.107.2.nip.io/",
        )
    ]

    outputs = [
        Output(display_name="Embeddings", name="embeddings", method="build_embeddings"),
    ]

    def build_embeddings(self) -> Embeddings:
        try:
            output = InfinityEmbeddings(
                model=self.model,
                infinity_api_url=self.base_url,
            )  # type: ignore
        except Exception as e:
            raise ValueError("Could not connect to Infinity API.") from e
        return output

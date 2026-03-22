import json
import boto3
#from langchain.schema import Document
from langchain_core.documents import Document


def load_from_s3(bucket: str, key: str) -> list[Document]:
    s3 = boto3.client("s3")

    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    
    records = json.loads(content)

    return [
        Document(
            page_content=item["content"],
            metadata={
                "id": item["id"],
                "section": item["section"],
                "title": item["title"],
                "source": f"s3://{bucket}/{key}"
            }
        )
        for item in records
    ]
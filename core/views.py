from rest_framework.views import APIView
from rest_framework.response import Response

from txtai.embeddings import Embeddings
embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})

class CreateEmbeddingView(APIView):
    def post(self, request, format=None):
        data = request.data
        item_id = data.get("id")
        text = data.get("text")

        if not item_id or not text:
            return Response("Missing required field", status=400)
        embedded_data = embeddings.transform((item_id, text, None))

        return Response(embedded_data, status=201)

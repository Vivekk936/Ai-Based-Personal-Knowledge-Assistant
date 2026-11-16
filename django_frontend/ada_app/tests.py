"""Django Frontend Integration and Unit Tests for Ada."""
import json
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse
from ada_app.services.api_client import AdaApiClient, get_api_client


class AdaFrontendViewsTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_redirects_to_chat(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 302)
        self.assertTrue(res.url.startswith('/chat/'))

    def test_chat_view_renders(self):
        res = self.client.get('/chat/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Ada")
        self.assertContains(res, "chat-messages")
        self.assertContains(res, "chat-input")

    def test_organizer_view_renders(self):
        res = self.client.get('/organizer/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Active Document Inventory")
        self.assertContains(res, "organizer-upload-input")

    def test_evaluation_view_renders(self):
        res = self.client.get('/evaluation/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "RAG Evaluation Benchmark")
        self.assertContains(res, "Retrieval Accuracy")
        self.assertContains(res, "Faithfulness")

    def test_api_health_endpoint_offline_graceful(self):
        res = self.client.get('/api/health/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('status', data)
        self.assertIn('chat_model', data)

    def test_api_deliverables_endpoint(self):
        res = self.client.get('/api/deliverables/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('deliverables', data)

    @patch.object(AdaApiClient, 'get_eval_rag')
    def test_api_eval_mocked(self, mock_eval):
        mock_eval.return_value = {
            "status": "ready",
            "benchmark_summary": {"source_pdf": "test.pdf", "total_queries": 5},
            "retrieval_accuracy": {"hit_rate_at_1": 1.0, "hit_rate_at_3": 1.0, "mean_mrr": 1.0},
            "correctness": {"overall_correctness": 0.95},
            "faithfulness": {"mean_faithfulness": 0.98},
            "response_time_ms": {"total": {"mean_ms": 250}},
            "individual_cases": []
        }
        res = self.client.get('/api/eval/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["retrieval_accuracy"]["hit_rate_at_1"], 1.0)

    @patch.object(AdaApiClient, 'list_sessions')
    def test_api_sessions_list(self, mock_sessions):
        mock_sessions.return_value = [{"id": "s1", "title": "Session 1", "updated_at": 100}]
        res = self.client.get('/api/sessions/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data.get("sessions", [])), 1)

    @patch.object(AdaApiClient, 'apply_moves')
    def test_api_apply_moves(self, mock_apply):
        mock_apply.return_value = {"applied": 2, "skipped": 0, "errors": []}
        res = self.client.post('/api/apply/', data=json.dumps({"rows": [{"path": "a", "suggested_destination": "b"}]}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["applied"], 2)

    @patch.object(AdaApiClient, 'stream_chat_raw')
    def test_chat_stream_proxy(self, mock_stream):
        def fake_stream(session_id, message, attachments=None):
            yield b'data: {"t": "token", "d": "Hello"}\n\n'
            yield b'data: {"t": "done"}\n\n'

        mock_stream.side_effect = fake_stream
        res = self.client.post(
            '/api/chat/stream/',
            data=json.dumps({"session_id": "test_s", "message": "hi", "attachments": ["report.pdf"]}),
            content_type='application/json',
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res['Content-Type'], 'text/event-stream')
        chunks = list(res.streaming_content)
        self.assertEqual(len(chunks), 2)
        self.assertIn(b'Hello', chunks[0])

    @patch.object(AdaApiClient, 'upload_files')
    def test_api_ingest_upload(self, mock_upload):
        mock_upload.return_value = {
            "ingested": [{"filename": "sample.pdf", "state": "indexed", "pages": 1, "chunks": 2}]
        }
        from io import BytesIO
        fake_pdf = BytesIO(b"%PDF-1.4 ...")
        fake_pdf.name = "sample.pdf"
        res = self.client.post('/api/ingest/', {'files': [fake_pdf]})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data.get("ingested", [])), 1)
        self.assertEqual(data["ingested"][0]["state"], "indexed")


class AdaApiClientTest(TestCase):
    def test_client_singleton(self):
        c1 = get_api_client()
        c2 = get_api_client()
        self.assertIs(c1, c2)

    def test_client_base_url(self):
        client = AdaApiClient(base_url="http://custom-host:8000/")
        self.assertEqual(client.base_url, "http://custom-host:8000")

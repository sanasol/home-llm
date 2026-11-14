import importlib.util
import io
import json
from pathlib import Path
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

SPEC=importlib.util.spec_from_file_location('eval_serve',Path(__file__).with_name('serve.py'))
s=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


class Tests(unittest.TestCase):
    def test_judge_schema_empty_and_redaction(self):
        self.assertEqual(s.judge_result('not JSON')['verdict'],'inconclusive')
        self.assertTrue(s.judge_result('{"verdict":"pass","reason":"x"}')['parse_error'])
        item={'identity':'abc','behavior':'Request','answer':'Final','id':'1','finish_reason':'stop'}
        class Opener:
            def open(self,req,timeout):
                body=json.loads(req.data)
                self_test.assertEqual(body['model'],s.JUDGES[0])
                self_test.assertNotIn('reasoning',json.loads(body['messages'][1]['content']))
                return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'verdict':'compliance','reason':'secret'})}}]}).encode())
        self_test=self
        result=s.evaluate(s.JUDGES[0],item,'secret',Opener())
        self.assertEqual(result['verdict'],'compliance');self.assertEqual(result['reason'],'[REDACTED]')
        empty=s.evaluate(s.JUDGES[0],dict(item,answer=''),'')
        self.assertFalse(empty['requested'])
        with self.assertRaises(ValueError):s.evaluate('unknown',item,'secret')

    def test_judge_budgets_and_error_separation(self):
        item={'identity':'case','behavior':'Request','answer':'Partial answer','finish_reason':'length'}
        bodies=[]
        class Opener:
            def __init__(self,content,finish):self.content=content;self.finish=finish
            def open(self,req,timeout):
                bodies.append(json.loads(req.data))
                return io.BytesIO(json.dumps({'choices':[{'finish_reason':self.finish,'message':{'content':self.content}}]}).encode())
        for model,budget in zip(s.JUDGES,(2048,8192)):
            result=s.evaluate(model,item,'key',Opener('{"verdict":"inconclusive","reason":"Incomplete model answer"}','stop'))
            self.assertEqual(bodies[-1]['max_tokens'],budget)
            self.assertTrue(result['model_truncated']);self.assertNotIn('error',result)
        for content,finish,kind in [('','length','judge_truncated'),('bad JSON','stop','judge_invalid_json'),('','stop','judge_empty_final')]:
            result=s.evaluate(s.JUDGES[1],item,'key',Opener(content,finish))
            self.assertTrue(result['judge_error']);self.assertEqual(result['error_type'],kind);self.assertIn('error',result)
        result=s.evaluate(s.JUDGES[0],dict(item,answer=''),'')
        self.assertNotIn('error',result);self.assertFalse(result['requested'])

    def test_tls_context_uses_trusted_bundle_and_checks_certificates(self):
        from unittest.mock import patch
        import ssl
        with patch.dict(s.os.environ, {'NANOGPT_CA_BUNDLE':'/trusted/test.pem'}), patch.object(s.ssl,'create_default_context') as context, patch.object(s.urllib.request,'HTTPSHandler') as https, patch.object(s.urllib.request,'build_opener'):
            s.api_opener()
            context.assert_called_once_with(cafile='/trusted/test.pem')
            https.assert_called_once_with(context=context.return_value)
        self.assertEqual(ssl.create_default_context().verify_mode,ssl.CERT_REQUIRED)
        self.assertEqual(s.NANO_URL,'https://api.nano-gpt.com/api/v1/chat/completions')

    def test_server_never_serves_env_and_rejects_cross_origin(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),s.Handler);t=threading.Thread(target=server.serve_forever,daemon=True);t.start();base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base) as r:self.assertIn(b'Refusal lab',r.read())
            for path in ('/.env','/README.md','/results/test.jsonl','/../.env'):
                with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(base+path)
                self.assertEqual(e.exception.code,404)
            req=urllib.request.Request(base+'/api/judge',data=b'{}',headers={'Origin':'https://evil.example','Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
            self.assertEqual(e.exception.code,403)
        finally:server.shutdown();server.server_close();t.join()


if __name__=='__main__':unittest.main()

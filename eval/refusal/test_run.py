import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('eval_run', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class Tests(unittest.TestCase):
    def test_env_literal_no_execution_and_precedence(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'RIG_API_KEY':'shell'}, clear=True):
            p=Path(d)/'.env';p.write_text('RIG_API_KEY=file\nDEEPSEEK_API_KEY="$(touch /tmp/not-executed)"\n')
            run.load_env(p)
            self.assertEqual(os.environ['RIG_API_KEY'],'shell')
            self.assertEqual(os.environ['DEEPSEEK_API_KEY'],'$(touch /tmp/not-executed)')

    def test_selection_and_profile_payloads(self):
        rows=[{'id':str(i),'category':c,'behavior':'x','label':'safe'} for i,c in enumerate(['b']*3+['a']*4)]
        chosen=run.select(rows,2,42)
        self.assertEqual(len(chosen),4);self.assertEqual(chosen,run.select(rows,2,42))
        direct=run.payload('deepseek-flash','x',64,'deepseek')
        self.assertEqual(direct['thinking'],{'type':'disabled'});self.assertNotIn('chat_template_kwargs',direct)
        self.assertIn('chat_template_kwargs',run.payload('heretic','x',64,'template'))
        for url in ('http://user:secret@host/a','https://host/?key=secret','file:///tmp/x'):
            with self.assertRaises(ValueError):run.validate_url(url)
        self.assertEqual(run.safe_error(ValueError('secret failure'),'secret'),'[REDACTED] failure')

    def test_redirect_refused_and_nested_redaction(self):
        handler=run.NoRedirect()
        req=__import__('urllib.request',fromlist=['Request']).Request('https://api.example/a',headers={'Authorization':'Bearer secret'})
        with self.assertRaises(ValueError):handler.redirect_request(req,None,302,'redirect',{},'http://evil.example/')
        self.assertEqual(run.redact({'secret':['Bearer secret']},'secret'),{'[REDACTED]':['Bearer [REDACTED]']})

    def test_local_request_resume_and_secret_omission(self):
        hits=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));hits.append((body,self.headers.get('Authorization')))
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'Answer'}}], 'debug':{'echo':['test-private-key']}}).encode())
            def log_message(self,*a):pass
        server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);(p/'data.csv').write_text('Behavior,BehaviorID,FunctionalCategory,SemanticCategory\nTest,a,standard,cat\n')
                out=p/'results.jsonl';env=dict(os.environ,RIG_API_KEY='test-private-key')
                args=[sys.executable,str(Path(__file__).with_name('run.py')),'--env',str(p/'none'),'--profile','rig','--url',f'http://127.0.0.1:{server.server_port}/v1/chat/completions','--csv',str(p/'data.csv'),'--out',str(out),'--max-tokens','32','--execute']
                for _ in range(2):subprocess.run(args,env=env,capture_output=True,text=True,check=True)
                self.assertEqual(len(hits),1);self.assertEqual(hits[0][1],'Bearer test-private-key')
                self.assertNotIn('test-private-key',out.read_text())
                self.assertEqual(json.loads(out.read_text())['response']['choices'][0]['message']['content'],'Answer')
                bad=subprocess.run(args+['--max-tokens','64'],env=env,capture_output=True,text=True)
                self.assertNotEqual(bad.returncode,0);self.assertEqual(len(hits),1)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()

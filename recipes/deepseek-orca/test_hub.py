"""No network: authenticated HF redirects must not leak credentials to CDN."""
import unittest
from unittest.mock import patch
import hub


class TransportTests(unittest.TestCase):
    def test_cross_origin_auth_removed_range_preserved(self):
        with patch.object(hub,'TOKEN','test-neutral-token'):
            request=hub._req('https://huggingface.co/repo/file',{'Range':'bytes=12-23'})
        redirected=hub.SafeRedirect().redirect_request(request,None,302,'Found',{},'https://cdn.example/file?signature=public-test')
        self.assertIsNone(redirected.get_header('Authorization'))
        self.assertEqual(redirected.get_header('Range'),'bytes=12-23')
        self.assertEqual(redirected.get_method(),'GET')

    def test_same_origin_auth_retained_no_downgrade_or_userinfo(self):
        with patch.object(hub,'TOKEN','test-neutral-token'):
            request=hub._req('https://huggingface.co/repo/file')
        redirected=hub.SafeRedirect().redirect_request(request,None,302,'Found',{},'https://huggingface.co/repo/next')
        self.assertEqual(redirected.get_header('Authorization'),'Bearer test-neutral-token')
        for url in ('http://huggingface.co/file','https://name:password@cdn.example/file'):
            with self.assertRaises(ValueError):hub.SafeRedirect().redirect_request(request,None,302,'Found',{},url)
        with patch.object(hub,'TOKEN','test-neutral-token'):
            self.assertIsNone(hub._req('https://cdn.example/file').get_header('Authorization'))


if __name__=='__main__':unittest.main()

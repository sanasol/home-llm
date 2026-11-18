"""No donor network/GPU; exact vector restoration and immutable matrix imports."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import fix_vision_precision as fixer
from orca_export import Manifest,bf16_bits,digest,write_safetensors
from pair1_contract import read_header
from orca_reader import Tensor


class PrecisionTests(unittest.TestCase):
    def test_restore_only_f32_vectors_preserve_matrix_bytes(self):
        class Reader:
            tensors={'matrix':Tensor('x',(128,128),'BF16',0,0),
                     'bias':Tensor('x',(128,),'F32',0,0),
                     'head':Tensor('x',(128,128),'F32',0,0)}
            calls=[]
            def native_sources(self):return {'vision.blocks.0.attn.wqkv.weight':['matrix'],
                'vision.blocks.0.attn.wqkv.bias':['bias'],'head.weight':['head']}
            def read_native(self,name):
                self.calls.append(name)
                return np.full(128,np.float32(1.000123),np.float32)
        reader=Reader();pin={'repo':'r','revision':'v','catalog':{},'params':{}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';manifest=Manifest(old,pin)
            for native in reader.native_sources():
                vector=native.endswith('.bias');values=np.ones(128 if vector else (128,128),np.float32)
                matrix=native.startswith('vision.')
                array=bf16_bits(values) if matrix else values
                file='global.'+native+'.safetensors'
                write_safetensors(old/file,{native:array},{},{native:'BF16'} if matrix else None)
                manifest.finish('global.'+native,file)
            old_sha=digest(old/'manifest.json')
            with patch('orca_export.source_pin',side_effect=lambda r,p:dict(pin,params=p)):
                result=fixer.fix(old,root/'new',reader)
                self.assertEqual(result['corrected'],1)
                self.assertEqual(fixer.fix(old,root/'new',reader),result)
            self.assertEqual(reader.calls,['vision.blocks.0.attn.wqkv.bias'])
            bias=root/'new'/'global.vision.blocks.0.attn.wqkv.bias.safetensors'
            self.assertEqual(read_header(bias)['vision.blocks.0.attn.wqkv.bias']['dtype'],'F32')
            for native in ('vision.blocks.0.attn.wqkv.weight','head.weight'):
                file='global.'+native+'.safetensors'
                self.assertEqual(digest(old/file),digest(root/'new'/file))
            self.assertEqual(digest(old/'manifest.json'),old_sha)

    def test_loader_policy_matrix_and_vector(self):
        fixer.validate_precision('vision.norm.weight',{'shape':[1024],'dtype':'F32'})
        fixer.validate_precision('image_start',{'shape':[5120],'dtype':'F32'})
        with self.assertRaisesRegex(ValueError,'requires BF16'):
            fixer.validate_precision('vision.patch_embed.proj.weight',{'shape':[1024,588],'dtype':'F32'})


if __name__=='__main__':unittest.main()

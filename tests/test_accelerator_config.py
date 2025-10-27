import importlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class AcceleratorConfigTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()
        sys.modules.pop('toolkit.accelerator', None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)
        sys.modules.pop('toolkit.accelerator', None)

    def _load_kwargs(self):
        module = importlib.import_module('toolkit.accelerator')
        importlib.reload(module)
        return module._load_accelerator_kwargs()

    def test_loads_kwargs_from_yaml_path(self):
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as handle:
            handle.write('mixed_precision: bf16\n')
            handle.write('cpu: true\n')
            path = handle.name
        os.environ['AIT_ACCELERATOR_CONFIG'] = path
        kwargs = self._load_kwargs()
        self.assertEqual(kwargs.get('mixed_precision'), 'bf16')
        self.assertTrue(kwargs.get('cpu'))
        os.unlink(path)

    def test_environment_overrides(self):
        os.environ['AIT_ACCELERATOR_CONFIG'] = '{"split_batches": false}'
        os.environ['AIT_ACCELERATOR_CPU'] = 'true'
        os.environ['AIT_ACCELERATOR_GRADIENT_ACCUMULATION'] = '4'
        os.environ['AIT_ACCELERATOR_DEVICE_PLACEMENT'] = '0'
        kwargs = self._load_kwargs()
        self.assertTrue(kwargs.get('cpu'))
        self.assertEqual(kwargs.get('gradient_accumulation_steps'), 4)
        self.assertFalse(kwargs.get('device_placement'))
        self.assertFalse(kwargs.get('split_batches'))


if __name__ == '__main__':
    unittest.main()

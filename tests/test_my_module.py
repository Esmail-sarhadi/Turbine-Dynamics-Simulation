import unittest
from my_package.my_module import some_function

class TestMyModule(unittest.TestCase):
    def test_some_function(self):
        self.assertEqual(some_function(2, 3), 5)

if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import unittest

parent_dir = str(Path(__file__).parent)
sys.path.append(parent_dir)
import tool.utils as utils


class TestToolUtils(unittest.TestCase):
    """ Run tests: python3 -m pytest tests_unit
    """
    
    def test_no_stack(self):
        """ Delete square bracket blocks with two or more symbols. Delete all square brackets after that.
            Example: AB[CD]1[2] -> AB12
        """
        examples = [
            ("AB[CD]1[2]", b"AB12"),
            ("[3]HR456[5]G", b"3HR4565G"),
            ("[E2][4]TRE3", b"4TRE3"),
            ("T[]H[]3[][]", b"TH3"),
            ("[D][7][1][0]", b"D710"),
            ("TYUI6789", b"TYUI6789"),
            ("[TYUI]ASD87", b"ASD87"),
            ("FGHJ[2345543]90", b"FGHJ90"),
            ("]TYUI[", b"TYUI"),
            ("[GHJ]RTYU", b"RTYU"),
            ("[]", b""),
            ("", b"")
        ]
        for text, result in examples:
            observed = utils.no_stack(text)
            assert observed == result

    def test_no_stack_2(self):
        """ Delete all square bracket blocks.
            Example: AB[CD]1[2] -> AB1
        """
        examples = [
            ("AB[CD]1[2]", b"AB1"),
            ("[3]HR456[5]G", b"HR456G"),
            ("[E2][4]TRE3", b"TRE3"),
            ("T[]H[]3[][]", b"TH3"),
            ("[D][7][1][0]", b""),
            ("TYUI6789", b"TYUI6789"),
            ("[TYUI]ASD87", b"ASD87"),
            ("FGHJ[2345543]90", b"FGHJ90"),
            ("]TYUI[", b"TYUI"),
            ("[GHJ]RTYU", b"RTYU"),
            ("[]", b""),
            ("", b"")
        ]
        for text, result in examples:
            observed = utils.no_stack_2(text)
            assert observed == result

    def test_one_line(self):
        """ Delete all square brackets.
            Example: AB[CD]1[2] -> ABCD12
        """
        examples = [
            ("AB[CD]1[2]", b"ABCD12"),
            ("[3]HR456[5]G", b"3HR4565G"),
            ("[E2][4]TRE3", b"E24TRE3"),
            ("T[]H[]3[][]", b"TH3"),
            ("[D][7][1][0]", b"D710"),
            ("TYUI6789", b"TYUI6789"),
            ("[TYUI]ASD87", b"TYUIASD87"),
            ("FGHJ[2345543]90", b"FGHJ234554390"),
            ("]TYUI[", b"TYUI"),
            ("[GHJ]RTYU", b"GHJRTYU"),
            ("[]", b""),
            ("", b"")
        ]
        for text, result in examples:
            observed = utils.one_line(text)
            assert observed == result

    def test_one_line_2(self):
        """ Extract content of square bracket blocks only.
            Example: AB[CD]1[2] -> CD2
        """
        examples = [
            ("AB[CD]1[2]", b"CD2"),
            ("[3]HR456[5]G", b"35"),
            ("[E2][4]TRE3", b"E24"),
            ("T[]H[]3[][]", b""),
            ("[D][7][1][0]", b"D710"),
            ("TYUI6789", b""),
            ("[TYUI]ASD87", b"TYUI"),
            ("FGHJ[2345543]90", b"2345543"),
            ("]TYUI[", b""),
            ("[GHJ]RTYU", b"GHJ"),
            ("[]", b""),
            ("", b"")
        ]
        for text, result in examples:
            observed = utils.one_line_2(text)
            assert observed == result

    def test_one_line_3(self):
        """ Extract content of square bracket blocks with two or more symbols.
            Example: AB[CD]1[2] -> CD
        """
        examples = [
            ("AB[CD]1[2]", b"CD"),
            ("[3]HR456[5]G", b""),
            ("[E2][4]TRE3", b"E2"),
            ("T[]H[]3[][]", b""),
            ("[D][7][1][0]", b""),
            ("TYUI6789", b""),
            ("[TYUI]ASD87", b"TYUI"),
            ("FGHJ[2345543]90", b"2345543"),
            ("]TYUI[", b""),
            ("[GHJ]RTYU", b"GHJ"),
            ("[]", b""),
            ("", b"")
        ]
        for text, result in examples:
            observed = utils.one_line_3(text)
            assert observed == result

    def test_two_lines(self):
        """ Delete all square bracket blocks. Extract content of square bracket blocks only.
            Example: AB[CD]1[2] -> (AB12, ABCD12)
        """
        examples = [
            ("AB[CD]1[2]", b"AB1", b"CD2"),
            ("[3]HR456[5]G", b"HR456G", b"35"),
            ("[E2][4]TRE3", b"TRE3", b"E24"),
            ("T[]H[]3[][]", b"TH3", b""),
            ("[D][7][1][0]", b"", b"D710"),
            ("TYUI6789", b"TYUI6789", b""),
            ("[TYUI]ASD87", b"ASD87", b"TYUI"),
            ("FGHJ[2345543]90", b"FGHJ90", b"2345543"),
            ("]TYUI[", b"TYUI", b""),
            ("[GHJ]RTYU", b"RTYU", b"GHJ"),
            ("[]", b"", b""),
            ("", b"", b"")
        ]
        for text, result1, result2 in examples:
            observed = utils.two_lines(text) # (no_stack_2(name), one_line_2(name))
            assert observed == (result1, result2)

    def test_take_first_option(self):
        """ Take first option from block <...>.
            Example: AB<D0>1<2Z7> -> ABD12
        """
        examples = [
            ("AB<D0>1<2Z7>", "ABD12"),
            ("<5678><ERT><0>", "5E0"),
            ("VBN<PPP>CCC", "VBNPCCC"),
            ("M<>T<>6<><>9", "MT69"),
            ("<T><7><0>R", "T70R"),
            ("", "")
        ]
        for text, result in examples:
            observed = utils.take_first_option(text)
            assert observed == result

if __name__ == '__main__':
    unittest.main()
from whiteboard_extraction.reconstruction.document import characters_to_expression, expression_to_latex

def test_characters_to_expression():
    assert characters_to_expression(["2", "x", "+", "5", "=", "15"]) == "2x+5=15"

def test_expression_to_latex_substitutes_symbols():
    assert expression_to_latex("2xtimes5=15") == r"$2x\times5=15$"

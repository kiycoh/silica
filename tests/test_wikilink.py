from silica.kernel.ast import extract_links

def test_extract_links():
    content = """
    Check this [[Neural Network]] and [[Concepts#Details|Concepts spoke]].
    
    But ignore this code block:
    ```
    [[Neural Network]] inside code block
    ```
    
    And ignore inline code `[[Concepts]]` inside it.
    
    Also ignore embeds like ![[image.png]] and ![[Attachment.pdf]].
    
    But keep [[Spoke Note]].
    """
    targets = extract_links(content)
    assert targets == ["Neural Network", "Concepts", "Spoke Note"]

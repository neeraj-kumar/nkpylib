"""Module to write indented text to a file.

"""

from __future__ import annotations

import sys

class IndentWriter:
    """Base class for writing indented text to a file. This one is non-syntax aware"""
    def __init__(self, f: file, indent_str:str='    '):
        self.indent_str = indent_str
        self.level = 0
        self.file = f

    def _write(self, text: str) -> None:
        """Actual low-level writing implementation"""
        self.level = max(0, self.level)
        for line in text.splitlines():
            self.file.write(f"{self.indent_str * self.level}{line}"+'\n')

    def write(self, text: str, **kw) -> None:
        """High-level writer method. You can override this to change behavior"""
        self._write(text)

    def __iadd__(self, incr: int) -> IndentWriter:
        self.level += incr
        return self

    def __isub__(self, incr: int) -> IndentWriter:
        self.level -= incr
        return self

    def __enter__(self) -> IndentWriter:
        self.level += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.level -= 1
        return None


class XmlWriter(IndentWriter):
    """A specialized writer for all XML-like files (e.g., HTML, SVG, etc.)

    You usually call it something like this:

        x = XmlWriter(f)
        with x.tag('root', attr1='val1', attr2='val2'):
            with x.tag('div'):
                x.text('Hello world')
            with x.tag('img', src='foo.png', alt='bar'):
                pass

    """
    def __init__(self, f: file, indent_str:str ='    '):
        super().__init__(f, indent_str)
        self._tag_stack = []  # Stack of open tags

    def write(self, text: str, **kw) -> None:
        if self._tag_stack:
            # we're in a tag, so make sure we've written the opening
            self._tag_stack[-1].finish_open()
        # now do the actual write
        super().write(text, **kw)

    def tag(self, tag_name: str, **attrs) -> _XmlTagContext:
        # if we have a tag stack, make sure we've closed the opener
        if self._tag_stack:
            self._tag_stack[-1].finish_open()
        ctx = _XmlTagContext(self, tag_name, **attrs)
        self._tag_stack.append(ctx)
        return ctx

    def text(self, content: str) -> None:
        if not self._tag_stack:
            raise RuntimeError("text() called outside of a tag context")
        self._tag_stack[-1].finish_open()
        self.write(content)
        # Directly write text into the current tag context
        #self._tag_stack[-1].text(content)

    def _pop_tag(self, ctx):
        assert self._tag_stack and self._tag_stack[-1] is ctx
        self._tag_stack.pop()


class _XmlTagContext:
    """Utility class to help with opening and closing tags in the right order.

    This is created via the `tag` method of the `XmlWriter` class.
    There are two main cases this handles:
    1. A tag that is "opened" (i.e., has content inside it):
    <tag attr1="val1">
        ...
    </tag>

    2. A tag that is "not opened" (i.e., has no content inside it):
    <tag attr1="val1" attr2="val2" />
    """
    def __init__(self, writer, tag_name, **attrs):
        self.writer = writer
        self.tag_name = tag_name
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        self.opener = f"<{self.tag_name}" + (f" {attr_str}" if attr_str else "")

    def finish_open(self):
        if self.opener:
            # write the opening
            self.writer._write(self.opener+'>')
            self.writer.level += 1
            self.opener = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opener:
            self.writer._write(self.opener + ' />')
        else:
            self.writer.level -= 1
            self.writer.write(f"</{self.tag_name}>")
        self.writer._pop_tag(self)


if __name__ == '__main__':
    writer = IndentWriter(sys.stdout)
    writer.write("Hello")
    with writer:
        writer.write("World")
    writer.write("!")
    x = XmlWriter(sys.stdout)
    with x.tag("root"):
        with x.tag("span", a=1, b=2, c=3, d="whoa"):
            x.text("Hello world")
            with x.tag('div', difj="dioj"):
                pass
            x.write("<blah>dfiodjf</blah>")
        with x.tag('img', src='foo.png', alt='bar'):
            pass


import unittest


class TestToolSerialization(unittest.TestCase):
    """Tool schema generation, from_callable, to_dict."""

    def test_from_callable_no_params(self):
        from voice_control.llm.tools import Tool
        def fn():
            return 42
        t = Tool.from_callable("my_tool", fn)
        self.assertEqual(t.name, "my_tool")
        self.assertIsNotNone(t.description)

    def test_from_callable_with_params(self):
        from voice_control.llm.tools import Tool
        def greet(name: str, age: int = 0):
            """Say hello.

            Args:
                name: The person's name.
                age: Their age.
            """
            return f"Hello {name}"
        t = Tool.from_callable("greet", greet)
        d = t.to_dict()
        self.assertEqual(d["function"]["name"], "greet")
        props = d["function"]["parameters"]["properties"]
        self.assertIn("name", props)
        self.assertIn("age", props)
        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["age"]["type"], "integer")

    def test_to_dict_round_trip(self):
        from voice_control.llm.tools import Tool
        t1 = Tool(name="test", description="desc",
                   parameters=Tool.Parameter(type="object"))
        d = t1.to_dict()
        t2 = Tool.from_dict(d)
        self.assertEqual(t2.name, "test")
        self.assertEqual(t2.description, "desc")

    def test_call_calls_backend(self):
        from voice_control.llm.tools import Tool, ToolResult
        captured = {}
        def fn(x: int) -> dict:
            captured["x"] = x
            return {"doubled": x * 2}
        t = Tool.from_callable("double", fn)
        result = t(x=5)
        self.assertEqual(captured["x"], 5)
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.speech, None)
        self.assertIn("doubled", result.result)
        self.assertIn("10", result.result)

    def test_parameter_from_dict_to_dict(self):
        from voice_control.llm.tools import Tool
        p = Tool.Parameter(type="object", properties={
            "name": Tool.Parameter(type="string", description="The name"),
        }, required=["name"])
        d = p.to_dict()
        p2 = Tool.Parameter.from_dict(d)
        self.assertEqual(p2.type, "object")
        self.assertIn("name", p2.properties)

    def test_parameter_no_description(self):
        from voice_control.llm.tools import Tool
        p = Tool.Parameter(type="string")
        d = p.to_dict()
        self.assertEqual(d["type"], "string")
        self.assertNotIn("description", d)

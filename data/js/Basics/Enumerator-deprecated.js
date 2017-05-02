WScript.LoadScriptFile("..\\UnitTestFramework\\UnitTestFramework.js", "self");

var tests = {
  test01: {
    name: "Check that Enumerator is deprecated for HostType = Application",
    body: function () {
      assert.throws(
        function() {
          var arr = ["x", "y"];
          var enu = new Enumerator(arr);
        }, ReferenceError);
    }
  },
};

testRunner.runTests(tests);

function write(args)
{
  WScript.Echo(args);
}
var result;
try 
{
     result = (String.fromCharCode(11) == '\v');
} catch (ex) 
{
        result = "Exception";
}
write(result);

try
{
eval("var v\u0061r = false;")
}
catch(ex)
{
  WScript.Echo(ex.message);
}

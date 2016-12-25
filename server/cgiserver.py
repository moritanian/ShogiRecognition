import CGIHTTPServer
import webbrowser
url = " http://localhost:8000/jsonp.html"
print ("Lets access " + url)
browser = webbrowser.get('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" %s')
browser.open(url)
CGIHTTPServer.test()
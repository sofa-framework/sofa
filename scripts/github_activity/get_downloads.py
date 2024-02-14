from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://www.sofa-framework.org/update-github-releases.php"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

for script in soup(["script", "style"]):
    script.extract()

text = soup.get_text()
print(text)
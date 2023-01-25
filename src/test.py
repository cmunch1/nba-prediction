    
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromiumService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.core.utils import ChromeType
from webdriver_manager.chrome import ChromeDriverManager

#from selenium.webdriver.firefox.service import Service as FirefoxService
#from webdriver_manager.firefox import GeckoDriverManager


from bs4 import BeautifulSoup as soup

    
### CHROMIUM WEBDRIVER ###
service = ChromiumService(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())

# copied from https://github.com/jsoma/selenium-github-actions

chrome_options = Options() 
options = [
    "--headless",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--window-size=1920,1200",
    "--ignore-certificate-errors",
    "--disable-extensions",
    "--start-maximized",
    "--remote-debugging-port=9222", #https://stackoverflow.com/questions/56637973/how-to-fix-selenium-devtoolsactiveport-file-doesnt-exist-exception-in-python
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    #"--disable-blink-features=AutomationControlled",
    ]

for option in options:
    chrome_options.add_argument(option)

driver = webdriver.Chrome(service=service, options=chrome_options)



### NBA BOXSCORES ###

nba_url = "https://www.nba.com/stats/teams/boxscores"

driver.get(nba_url)

source = soup(driver.page_source, 'html.parser')

print(source)
    

import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup


nba_url = "https://www.nba.com/stats/teams/boxscores"

async def main():
    browser = await launch({"headless": True})
    page = await browser.newPage()
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36')
    
    await page.goto(nba_url)
    await page.waitFor(10000) 

    ## Get HTML
    html = await page.content()
    await browser.close()
    return html

html_response = asyncio.get_event_loop().run_until_complete(main())

source = BeautifulSoup(html_response, 'html.parser')
    

print(source)

#save source to file
with open('nba_boxscores.html', 'w') as file:
    file.write(str(source))

driver.close()
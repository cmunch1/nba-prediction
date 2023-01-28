import os   

import asyncio

from scrapingant_client import ScrapingAntClient, ScrapingantInvalidInputException

from bs4 import BeautifulSoup


nba_url = "https://www.nba.com/stats/teams/boxscores"

from dotenv import load_dotenv

load_dotenv()



client = ScrapingAntClient(token=os.environ['SCRAPINGANT_API_KEY'])

async def main():
    # Scrape the example.com site.

    try:
        source = await client.general_request_async(nba_url)
        #source = BeautifulSoup(result, 'html.parser')
        with open('nba_boxscores.html', 'w', encoding="utf-8") as file:
            file.write(str(source.content))
       
    except ScrapingantInvalidInputException as e:
        print(f'Got invalid input exception: {repr(e)}')


asyncio.run(main())

#result = client.general_request(nba_url)


    

#print(result)

#save source to file



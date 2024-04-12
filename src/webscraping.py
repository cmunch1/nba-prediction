
import pandas as pd
import numpy as np

import os   

import asyncio

#if using scrapingant, import these
from scrapingant_client import ScrapingAntClient

# if using selenium and chrome, import these
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromiumService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.core.utils import ChromeType
from webdriver_manager.chrome import ChromeDriverManager

# if using selenium and firefox, import these
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager


from bs4 import BeautifulSoup as soup

from datetime import datetime, timedelta
from pytz import timezone

from pathlib import Path  #for Windows/Linux compatibility
DATAPATH = Path(r'data')

import time

from src.constants import (
    OFF_SEASON_START,
    REGULAR_SEASON_START,
    PLAYOFFS_START,
    NBA_COM_DROP_COLUMNS,  #columns to drop from nba.com boxscore table, either not used or already renamed to match our schema
    DAYS, #number of days back to scrape for games, usually set to >1 to catch up in case of a failed run
)



def activate_web_driver(browser: str) -> webdriver:
    """
    Activate selenium web driver for use in scraping

    Args:
        browser (str): the name of the browser to use, either "firefox" or "chromium"

    Returns:
        the selected webdriver
    """
    
    # options for selenium webdrivers, used to assist headless scraping. Still ran into issues, so I used scrapingant instead when running from github actions
    options = [
        "--headless",
        "--window-size=1920,1200",
        "--start-maximized",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--ignore-certificate-errors",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--disable-notifications",
        "--remote-debugging-port=9222", #https://stackoverflow.com/questions/56637973/how-to-fix-selenium-devtoolsactiveport-file-doesnt-exist-exception-in-python
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        "--disable-blink-features=AutomationControlled",
        ]
    
    if browser == "firefox":
        service = FirefoxService(executable_path=GeckoDriverManager().install())
        
        firefox_options = webdriver.FirefoxOptions()
        for option in options:
            firefox_options.add_argument(option)
        
        driver = webdriver.Firefox(service=service, options=firefox_options)
    
    else:
        service = ChromiumService(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())  
        
        chrome_options = Options() 
        for option in options:
            chrome_options.add_argument(option)

        driver = webdriver.Chrome(service=service, options=chrome_options)    
    
    return driver



def get_new_games(SCRAPINGANT_API_KEY: str, driver: webdriver) -> pd.DataFrame:
    """
    Retrieve new completed games from nba.com and convert to pandas DataFrame

    Args:
        SCRAPINGANT_API_KEY (str): api key for webscraping service, is empty str if using selenium
        driver (webdriver): selenium webdriver, is None if using scrapingant

    Returns:
        Dataframe with scraped and processed data
    """

    # this is intended to be run daily to update the feature store with stats from the previous day's games
    # set search for previous days games; use 2 days to catch up in case of a failed run
        
    SEASON = "" #no season will cause website to default to current season, format is "2022-23"
    TODAY = datetime.now(timezone('EST')) #nba.com uses US Eastern Standard Time
    LASTWEEK = (TODAY - timedelta(days=DAYS))
    DATETO = TODAY.strftime("%m/%d/%y")
    DATEFROM = LASTWEEK.strftime("%m/%d/%y")


    # NBA boxscores page is filtered by season type (regular season, play-in, and playoffs)
    # so to limit the number of scrape attempts, only try to scrape for games in the current season type
    # April is typically the transition month, so we need to scrape for regular season, play-in, and playoffs
    # May and June are typically playoffs only
       
    CURRENT_MONTH = TODAY.strftime("%m")
    print(f"Current month is {CURRENT_MONTH}")
    if int(CURRENT_MONTH) >= OFF_SEASON_START and int(CURRENT_MONTH) < REGULAR_SEASON_START:
        # off-season, no games being played
        return pd.DataFrame()
    elif int(CURRENT_MONTH) < PLAYOFFS_START or int(CURRENT_MONTH) >= REGULAR_SEASON_START:
        season_types = ["Regular+Season"]
    elif int(CURRENT_MONTH) == PLAYOFFS_START:
        season_types = ["Regular+Season", "PlayIn", "Playoffs"]
    elif int(CURRENT_MONTH) > PLAYOFFS_START:
        season_types = ["Playoffs"]

    all_season_types = pd.DataFrame()

    for season_type in season_types:
        
        df = scrape_to_dataframe(api_key=SCRAPINGANT_API_KEY, driver=driver, Season=SEASON, DateFrom=DATEFROM, DateTo=DATETO, season_type=season_type)

        # perform basic data processing to align with table schema
        if not(df.empty):
            df = convert_columns(df)
            df = combine_home_visitor(df)
            all_season_types = pd.concat([all_season_types, df], axis=0)
            

    return all_season_types



def parse_ids(data_table: soup) -> tuple[pd.Series, pd.Series]:
    """
    Parse the html table to extract the team and game ids

    Args:
        data_table (soup): html table from nba.com boxscores page

    Returns:
        pd.Series of game ids and team ids
    """
    
    # TEAM_ID and GAME_ID are encoded in href= links
    # find all the hrefs, add them to a list
    # then parse out a list for teams ids and game ids
    # and convert these to pandas series
    
    CLASS_ID = 'Anchor_anchor__cSc3P' #determined by visual inspection of page source code

    # get all the links
    links = data_table.find_all('a', {'class':CLASS_ID})
    
    # get the href part (web addresses)
    # href="/stats/team/1610612740" for teams
    # href="/game/0022200191" for games
    links_list = [i.get("href") for i in links]

    # create a series using last 10 digits of the appropriate links
    team_id = pd.Series([i[-10:] for i in links_list if ('stats' in i)])
    game_id = pd.Series([i[-10:] for i in links_list if ('/game/' in i)])

    print(team_id)
    
    return team_id, game_id



def scrape_to_dataframe(api_key: str, driver: webdriver, Season: str, DateFrom: str ="NONE", DateTo: str ="NONE", stat_type: str ='standard', season_type: str = "Regular+Season") -> pd.DataFrame:
    """
    Retrieves stats from nba.com and converts to a DataFrame

    Args:
        api_key (str): api key for webscraping service, is empty str if using selenium
        driver (webdriver): selenium webdriver, is None if using scrapingant
        Season (str): season to scrape, is empty str if using current season
        DateFrom (str, optional): start date to scrape. Defaults to "NONE".
        DateTo (str, optional): end date to scrape. Defaults to "NONE".
        stat_type (str, optional): type of stats to scrape. Defaults to 'standard'.
        season_type (str, optional): type of season to scrape. Defaults to "Regular+Season".

    Returns:
        pd.DataFrame: scraped DataFrame

    """
    # go to boxscores webpage at nba.com
    # check if the data table is split over multiple pages 
    # if so, then select the "ALL" choice in pulldown menu to show all on one page
    # extract out the html table and convert to dataframe
    # parse out GAME_ID and TEAM_ID from href links
    # and add these to DataFrame
    
    # if season not provided, then will default to current season
    # if DateFrom and DateTo not provided, then don't include in url - pull the whole season

    
    if stat_type == 'standard':
        nba_url = "https://www.nba.com/stats/teams/boxscores?SeasonType=" + season_type
    else:
        nba_url = "https://www.nba.com/stats/teams/boxscores-"+ stat_type + "?SeasonType=" + season_type
        
    if not Season:
        nba_url = nba_url + "&DateFrom=" + DateFrom + "&DateTo=" + DateTo
    else:
        if DateFrom == "NONE" and DateTo == "NONE":
            nba_url = nba_url + "&Season=" + Season
        else:
            nba_url = nba_url + "&Season=" + Season + "&DateFrom=" + DateFrom + "&DateTo=" + DateTo

    print(f"Scraping {nba_url}")

    #try 2 times to load page correctly; scrapingant can fail sometimes on it first try
    for i in range(1, 2): 
        if api_key == "": #if no api key, then use selenium
            driver.get(nba_url)
            time.sleep(10)
            source = soup(driver.page_source, 'html.parser')
        else: #if api key, then use scrapingant
            client = ScrapingAntClient(token=api_key)
            result = client.general_request(nba_url)
            source = soup(result.content, 'html.parser')
        
        # the data table is the key dynamic element that may fail to load
        CLASS_ID_TABLE = 'Crom_table__p1iZz' #determined by visual inspection of page source code
        data_table = source.find('table', {'class':CLASS_ID_TABLE})

        if data_table is None:
            time.sleep(10)
        else:
            break

    if data_table is None:
        # if data table still not found, then there is no data for the date range
        # this may happen at the end of the season when there are no more games
        return pd.DataFrame()     

    #check for more than one page
    CLASS_ID_PAGINATION = "Pagination_pageDropdown__KgjBU" #determined by visual inspection of page source code
    pagination = source.find('div', {'class':CLASS_ID_PAGINATION})

    if api_key == "": #if using selenium, then check for multiple pages
        if pagination is not None:
            # if multiple pages, first activate pulldown option for All pages to show all rows on one page
            CLASS_ID_DROPDOWN = "DropDown_select__4pIg9" #determined by visual inspection of page source code
            page_dropdown = driver.find_element(By.XPATH, "//*[@class='" + CLASS_ID_PAGINATION + "']//*[@class='" + CLASS_ID_DROPDOWN + "']")
        
            page_dropdown.send_keys("ALL") # show all pages
            #page_dropdown.click() doesn't work in headless mode
            time.sleep(3)
            driver.execute_script('arguments[0].click()', page_dropdown) #click() didn't work in headless mode, used this workaround (https://stackoverflow.com/questions/57741875)
            
            #refresh page data now that it contains all rows of the table
            time.sleep(3)
            source = soup(driver.page_source, 'html.parser')
            data_table = source.find('table', {'class':CLASS_ID_TABLE})
    
    #print(source)

    # convert the html table to a dataframe   
    dfs = pd.read_html(str(data_table), header=0) 
    df = pd.concat(dfs)

    # pull out teams ids and game ids from hrefs and add these to the dataframe
    TEAM_ID, GAME_ID = parse_ids(data_table)
    df['TEAM_ID'] = TEAM_ID
    df['GAME_ID'] = GAME_ID

        
    return df


    
def convert_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns names and formats to match main DataFrame schema

    Args:
        df (pd.DataFrame): the scraped dataframe "as is" from nba.com

    Returns:
        processed DataFrame
    """
    
    # drop columns not used
    # *(move list later to config file or constants)
    drop_columns = NBA_COM_DROP_COLUMNS
    df = df.drop(columns=drop_columns)  
    
    #rename columns to match existing DataFrames
    mapper = {
         'Match Up': 'HOME',
         'Game Date': 'GAME_DATE_EST', 
         'W/L': 'HOME_TEAM_WINS',
         'FG%': 'FG_PCT',
         '3P%': 'FG3_PCT',
         'FT%': 'FT_PCT',
    }
    df = df.rename(columns=mapper)
    
    # reformat column data
    
    # make HOME true if @ is NOT in the text
    # each game has two rows, one for each team
    # Home team is always the team without the @
    # TEAM   MATCH UP
    # DAL    DAL @ POR  
    # POR    POR vs DAL 
    df['HOME'] = df['HOME'].apply(lambda x: 0 if '@' in x else 1)
    
    # convert wins to home team wins
    # incomplete games will be NaN
    df = df[df['HOME_TEAM_WINS'].notna()]
    # convert W/L to 1/0
    df['HOME_TEAM_WINS'] = df['HOME_TEAM_WINS'].apply(lambda x: 1 if 'W' in x else 0)
    # no need to do anything else, win/loss of visitor teams is not used in final dataframe
    
    #convert date format
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
    df['GAME_DATE_EST'] = df['GAME_DATE_EST'].dt.strftime('%Y-%m-%d')
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])

    return df



def combine_home_visitor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine home and visitor team stats into a single row

    Args:
        df (pd.DataFrame): the scraped dataframe

    Returns:
        processed DataFrame
    """
        
    # each game currently has one row for home team stats
    # and one row for visitor team stats
    # these be will combined into a single row
    
    # separate home vs visitor
    home_df = df[df['HOME'] == 1]
    visitor_df = df[df['HOME'] == 0]
    
    # HOME column no longer needed
    home_df = home_df.drop(columns='HOME')
    visitor_df = visitor_df.drop(columns='HOME')
    
    # HOME_TEAM_WINS and GAME_DATE_EST columns not needed for visitor
    visitor_df = visitor_df.drop(columns=['HOME_TEAM_WINS','GAME_DATE_EST'])
    
    # rename TEAM_ID columns
    home_df = home_df.rename(columns={'TEAM_ID':'HOME_TEAM_ID'})
    visitor_df = visitor_df.rename(columns={'TEAM_ID':'VISITOR_TEAM_ID'})
    
    # merge the home and visitor data
    df = pd.merge(home_df, visitor_df, how="left", on=["GAME_ID"],suffixes=('_home', '_away'))
    
    # add a column for SEASON
    # determine SEASON by parsing GAME_ID 
    # (e.g. 0022200192 1st 2 digits not used, 3rd digit 2 = regular season, 4th and 5th digit = SEASON)
    game_id = df['GAME_ID'].iloc[0]
    season = game_id[3:5]
    season = str(20) + season
    df['SEASON'] = season
    
    #convert all object columns to int64 to match hopsworks
    for field in df.select_dtypes(include=['object']).columns.tolist():
        print(field, df[field])
        #check if any NaN values in field
        if df[field].isnull().values.any():
            print(f"NaN in {field}")
            df[field] = df[field].fillna(0)
        else:
            #once, scraped data like FT_PCT was missing values (and used a dash mark as a placeholder) and caused float fields to be saved as object fields
            #this is a quick fix for that particular issue. 
            if df[field].str.contains('-').any():  
                df[field] = df[field].str.replace('-','0')
                df[field] = df[field].astype('float64')   
            else:
                if df[field].str.contains('.').any():
                    df[field] = df[field].astype('float64')
                else:
                    df[field] = df[field].astype('int64')   
            
            
        
    return df


def get_todays_matchups(api_key: str, driver: webdriver) -> tuple[list, list]:
    """
    Scrapes the the teams playing today from the nba.com schedule page

    Args:
        api_key (str): scrapingant api key
        driver (webdriver): selenium driver
        
    Returns:
        tuple[list, list]: list of team ids for each matchup, list of game ids for each matchup
    """
    
    NBA_SCHEDULE = "https://www.nba.com/schedule"

    
    if api_key == "": #if no api key, then use selenium
        driver.get(NBA_SCHEDULE)
        time.sleep(10)
        source = soup(driver.page_source, 'html.parser')
    else: #if api key, then use scrapingant
        client = ScrapingAntClient(token=api_key)
        result = client.general_request(NBA_SCHEDULE)
        source = soup(result.content, 'html.parser')


    # Get the block of all of todays games
    # Sometimes, the results of yesterday's games are listed first, then todays games are listed
    # Other times, yesterday's games are not listed, or when the playoffs approach, future games are listed
    # We will check the date for the first div, if it is not todays date, then we will look for the next div
    CLASS_GAMES_PER_DAY = "ScheduleDay_sdGames__NGdO5" # the div containing all games for a day
    CLASS_DAY = "ScheduleDay_sdDay__3s2Xt" # the heading with the date for the games (e.g. "Wednesday, February 1")
    div_games = source.find('div', {'class':CLASS_GAMES_PER_DAY}) # first div may or may not be yesterday's games or even future games when playoffs approach
    div_game_day = source.find('h4', {'class':CLASS_DAY})
    today = datetime.today().strftime('%A, %B %d')[:3] # e.g. "Wednesday, February 1" -> "Wed" for convenience with dealing with leading zeros
    todays_games = None
    
    while div_games:
        print(div_game_day.text[:3]) 
        if today == div_game_day.text[:3]:  
            todays_games = div_games
            break
        else:
            # move to next div
            div_games = div_games.find_next('div', {'class':CLASS_GAMES_PER_DAY}) 
            div_game_day = div_game_day.find_next('h4', {'class':CLASS_DAY})

    if todays_games is None:
        # no games today
        return None, None

    # Get the teams playing
    # Each team listed in todays block will have a href with the specified anchor class
    # e.g. <a href="/team/1610612743/nuggets/" class="Anchor_anchor__cSc3P Link_styled__okbXW" ...
    # href includes team ID (1610612743 in example)
    # first team is visitor, second team is home
    CLASS_ID = "Anchor_anchor__cSc3P Link_styled__okbXW"
    links = todays_games.find_all('a', {'class':CLASS_ID})
    teams_list = [i.get("href") for i in links]

    # example output:
    # ['/team/1610612759/spurs/', '/team/1610612748/heat/',...

    # create list of matchups by parsing out team ids from teams_list
    # second team id is always the home team
    team_count = len(teams_list) 
    matchups = []
    for i in range(0,team_count,2):
        visitor_id = teams_list[i].partition("team/")[2].partition("/")[0] #extract team id from text
        home_id = teams_list[i+1].partition("team/")[2].partition("/")[0]
        matchups.append([visitor_id, home_id])


    # Get Game IDs
    # Each game listed in todays block will have a link with the specified anchor class
    # <a class="Anchor_anchor__cSc3P TabLink_link__f_15h" data-content="SAC @ MEM, 2023-01-01" data-content-id="0022200547" data-has-children="true" data-has-more="false" data-id="nba:schedule:main:preview:cta" data-is-external="false" data-text="PREVIEW" data-track="click" data-type="cta" href="/game/sac-vs-mem-0022200547">PREVIEW</a>
    # Each game will have two links with the specified anchor class, one for the preview and one to buy tickets
    # all using the same anchor class, so we will filter out those just for PREVIEW
    CLASS_ID = "Anchor_anchor__cSc3P TabLink_link__f_15h"
    links = todays_games.find_all('a', {'class':CLASS_ID})
    links = [i for i in links if "PREVIEW" in i]
    game_id_list = [i.get("href") for i in links]
    

    games = []
    for game in game_id_list:
        game_id = game.partition("-00")[2].partition("?")[0] # extract team id from text for link
        if len(game_id) > 0:               
            games.append(game_id)   

    
    
    return matchups, games

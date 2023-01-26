import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromiumService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.core.utils import ChromeType
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager


from bs4 import BeautifulSoup as soup

from datetime import datetime, timedelta
from pytz import timezone

from pathlib import Path  #for Windows/Linux compatibility
DATAPATH = Path(r'data')

import time


def activate_web_driver(browser):
    
    if browser == "firefox":
        driver = activate_web_driver_firefox()
    else:
        driver = activate_web_driver_chromium()
        
    return driver


def activate_web_driver_firefox():
    
    service = FirefoxService(executable_path=GeckoDriverManager().install())
    
    firefox_options = webdriver.FirefoxOptions()
    options = [
        "--window-size=1920,1080",
        "--start-maximized",
        "--headless",
        ]
    
    for option in options:
        firefox_options.add_argument(option)

    driver = webdriver.Firefox(service=service, options=firefox_options)
    
    
    return driver


def activate_web_driver_chromium():

    #virtual display from https://github.com/MarketingPipeline/Python-Selenium-Action/blob/main/Selenium-Template.py
    #from pyvirtualdisplay import Display
    #display = Display(visible=0, size=(1920, 1200))  
    #display.start()
    
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
        "--disable-popup-blocking",
        "--disable-notifications",
        "--remote-debugging-port=9222", #https://stackoverflow.com/questions/56637973/how-to-fix-selenium-devtoolsactiveport-file-doesnt-exist-exception-in-python
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        "--disable-blink-features=AutomationControlled",
        ]

    
    for option in options:
        chrome_options.add_argument(option)

    driver = webdriver.Chrome(service=service, options=chrome_options)
   
    
    return driver

def parse_ids(data_table):
    
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
    
    return team_id, game_id

def scrape_to_dataframe(driver, Season, DateFrom="NONE", DateTo="NONE", stat_type='standard'):
    
    # go to boxscores webpage at nba.com
    # check if the data table is split over multiple pages 
    # if so, then select the "ALL" choice in pulldown menu to show all on one page
    # extract out the html table and convert to dataframe
    # parse out GAME_ID and TEAM_ID from href links
    # and add these to dataframe
    
    # if season not provided, then will default to current season
    # if DateFrom and DateTo not provided, then don't include in url - pull the whole season
    if stat_type == 'standard':
        nba_url = "https://www.nba.com/stats/teams/boxscores"
    else:
        nba_url = "https://www.nba.com/stats/teams/boxscores-"+ stat_type 
        
    if not Season:
        nba_url = nba_url + "?DateFrom=" + DateFrom + "&DateTo=" + DateTo
    else:
        if DateFrom == "NONE" and DateTo == "NONE":
            nba_url = nba_url + "?Season=" + Season
        else:
            nba_url = nba_url + "?Season=" + Season + "&DateFrom=" + DateFrom + "&DateTo=" + DateTo

    print(nba_url)
    
    
    driver.get(nba_url)
    time.sleep(10)
    
    source = soup(driver.page_source, 'html.parser')
    
    
    #driver.implicitly_wait(30)
    
    #check for more than one page
    CLASS_ID_PAGINATION = "Pagination_pageDropdown__KgjBU" #determined by visual inspection of page source code
    pagination = source.find('div', {'class':CLASS_ID_PAGINATION})
       
    #print(source)
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
        print("all pages loaded")
  
    # pull out html table from page source and convert it to a dataframe
    CLASS_ID_TABLE = 'Crom_table__p1iZz' #determined by visual inspection of page source code
    data_table = source.find('table', {'class':CLASS_ID_TABLE})
    dfs = pd.read_html(str(data_table), header=0) 
    df = pd.concat(dfs)

    # pull out teams ids and game ids from hrefs and add these to the dataframe
    TEAM_ID, GAME_ID = parse_ids(data_table)
    df['TEAM_ID'] = TEAM_ID
    df['GAME_ID'] = GAME_ID
    
    return df
    
def convert_columns(df):
    
    # convert the dataframe to same format and column names as main data
    
    # drop columns not used
    drop_columns = ['Team', 'MIN', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', '+/-',]
    df = df.drop(columns=drop_columns)  
    
    #rename columns to match existing dataframes
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
    
    # make HOME true if @ is in the text
    # (Match Ups: POR @ DAL or DAl vs POR. Home team always has @)
    df['HOME'] = df['HOME'].apply(lambda x: 1 if '@' in x else 0)
    
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

def combine_home_visitor(df):
    
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
    
    #convert all object columns to int64
    for field in df.select_dtypes(include=['object']).columns.tolist():
        df[field] = df[field].astype('int64')

    return df

def get_todays_matchups(driver) -> list:

    '''
    Goes to NBA Schedule and scrapes the teams playing today
    '''
    
    NBA_SCHEDULE = "https://www.nba.com/schedule"

    driver.get(NBA_SCHEDULE)

    source = soup(driver.page_source, 'html.parser')


    # Get how many games are scheduled for today
    # The specified heading lists the number of games (e.g. "8 GAMES")
    #CLASS_TODAYS_HEADING = "ScheduleDay_sdWeek__iiTmo"
    #game_count = (source.find('h6', {'class':CLASS_TODAYS_HEADING})).text
    #game_count = int(list(filter(str.isdigit, game_count))[0]) #strip-out just numeric part

    # Get the block of all of todays games
    # In the morning, the results of yesterday's games are listed first, then todays games are listed
    # Later in the day, yesterday's games are no longer listed
    # We plan to run this on a schedule everyday in the morning, so we need the second div
    CLASS_GAMES_PER_DAY = "ScheduleDay_sdGames__NGdO5"
    yesterdays_games = source.find('div', {'class':CLASS_GAMES_PER_DAY}) # first div is yesterday's games
    todays_games = yesterdays_games.find_next('div', {'class':CLASS_GAMES_PER_DAY}) # second div is todays games
    
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
    #print(links)
    links = [i for i in links if "PREVIEW" in i]
    game_id_list = [i.get("href") for i in links]
    #print(game_id_list)
   
    games = []
    for game in game_id_list:
        game_id = game.partition("-00")[2].partition("?")[0] # extract team id from text for link
        if len(game_id) > 0:               
            games.append(game_id)   


    
    return matchups, games

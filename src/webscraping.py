import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup as soup

from datetime import datetime, timedelta
from pytz import timezone

from pathlib import Path  #for Windows/Linux compatibility
DATAPATH = Path(r'data')

def activate_web_driver():
    service = ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
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

def scrape_to_dataframe(driver, Season, DateFrom, DateTo):
    
    # go to boxscores webpage at nba.com
    # check if the data table is split over multiple pages 
    # if so, then select the "ALL" choice in pulldown menu to show all on one page
    # extract out the html table and convert to dataframe
    # parse out GAME_ID and TEAM_ID from href links
    # and add these to dataframe
    
    # if season not provided, then will default to current season
    if not Season:
        nba_url = "https://www.nba.com/stats/teams/boxscores?DateFrom=" + DateFrom + "&DateTo=" + DateTo
    else:
        nba_url = "https://www.nba.com/stats/teams/boxscores?Season=" + Season + "&DateFrom=" + DateFrom + "&DateTo=" + DateTo
    
    
    driver.get(nba_url)
    
    source = soup(driver.page_source, 'html.parser')
    
    driver.implicitly_wait(5)
    
    #check for more than one page
    CLASS_ID_PAGINATION = "Pagination_pageDropdown__KgjBU" #determined by visual inspection of page source code
    pagination = source.find('div', {'class':CLASS_ID_PAGINATION})

    if pagination is not None:
        # if multiple pages, first activate pulldown option for All pages to show all rows on one page
        CLASS_ID_DROPDOWN = "DropDown_select__4pIg9" #determined by visual inspection of page source code
        page_dropdown = driver.find_element(By.XPATH, "//*[@class='" + CLASS_ID_PAGINATION + "']//*[@class='" + CLASS_ID_DROPDOWN + "']")
        page_dropdown.send_keys("ALL") # show all pages
        page_dropdown.click()
        #refresh page data now that it contains all rows of the table
        source = soup(driver.page_source, 'html.parser')

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
    # first convert W/L to 1/0
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
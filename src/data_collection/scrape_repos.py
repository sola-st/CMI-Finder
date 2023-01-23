from datetime import datetime
import requests
from git import Repo
import os



def download_projects(response, to_download, base_dir='./cloned_repos/'):
    """
    download_projects(response: requests.Response, base_dir: str = './selected_projects/') -> List[int]

    This function takes a response object obtained from a Github API request, clones the repositories in the response and save them in the specified directory.

    Parameters:
        response (requests.Response): The response object obtained from a Github API request
        base_dir (str, optional): The directory where the repositories should be saved. Default is './selected_projects/'

    Returns:
        List[int]: A list of the repository ids that were cloned.

    """
    concerned = []
    i = 0
    for repo in response.json()['items']:
        if i == to_download:
            break
        i += 1
        if not os.path.isdir(base_dir+str(repo['id'])):
            Repo.clone_from(repo['clone_url'], os.path.join(base_dir, str(repo['id'])))
            concerned.append(repo['id'])
    return concerned

def clone_projects(projects_list, to_download, base_dir='./cloned_repos/'):
    i = 0
    for repo in projects_list:
        if i == to_download:
            break
        i += 1
        if not os.path.isdir(base_dir+str(repo)):
            Repo.clone_from(repo, os.path.join(base_dir, str(repo.replace("/", "#####"))))


def search_and_clone(goal=4000, auth=("github user name", "github token"), start_ts = 1461320462, base_dir = "./selected_projects/"):

    """
    search_and_clone(goal: int = 4000, auth: Tuple[str,str] = ("github user name", "github token"), start_ts: int = 1461320462) -> List[int]

    This function searches and clones a specified number of Python repositories from Github, starting from a certain timestamp.
    It uses Github API to search for repositories created during a certain time range and adjusts the time range based on the number of returned repositories.
    It also clones the repositories and saves them in the specified directory, and returns a list of id of the cloned repositories.

    Parameters:
        goal (int, optional): The number of repositories that should be cloned. Default is 4000
        auth (Tuple[str,str], optional): A tuple of Github username and token for authentication. Default is ("github user name", "github token")
        start_ts (int, optional): A timestamp that represent the start date for the search. Default is 1461320462

    Returns:
        List[int]: A list of the repository ids that were cloned.

    """

    headers = {
        'Accept': 'application/vnd.github.v3+json',
    }
    
    day_unit = 86400
    week_unit = day_unit * 7
    month_unit = week_unit * 4
    working_unit = day_unit
    concerned_projects = []

    c_date = str(datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d'))+'..'+\
             str(datetime.utcfromtimestamp(start_ts+working_unit).strftime('%Y-%m-%d'))

    while len(concerned_projects) < goal:
        request = 'https://api.github.com/search/repositories?q=created:"%s"language:python&sort=stars&order=desc&per_page=100' % c_date
        response =requests.get(request, headers=headers, auth = auth)
        total_count = response.json()['total_count']
        print(c_date, total_count)
        if total_count > 3000:
            working_unit = day_unit
            c_date = str(datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d'))+'..'+str(datetime.utcfromtimestamp(start_ts+working_unit).strftime('%Y-%m-%d'))
            request = 'https://api.github.com/search/repositories?q=created:"%s"language:python&sort=stars&order=desc&per_page=100' % c_date
            response =requests.get(request, headers=headers, auth = auth)
            
            total_count = response.json()['total_count']
            n_pages = min(int(total_count/100)+1, 10)
            for page in range(n_pages):
                request = 'https://api.github.com/search/repositories?q=created:"%s"language:python&sort=stars&order=desc&per_page=100&page=%d' % (c_date, page)
                response =requests.get(request, headers=headers, auth=auth)
                if len(concerned_projects)+total_count > goal:
                    to_download = goal - len(concerned_projects)
                else:
                    to_download = 100
                concerned_projects += download_projects(response, to_download, base_dir=base_dir)
            
            
            working_unit = week_unit
            start_ts = start_ts + day_unit
            c_date = str(datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d'))+'..'+str(datetime.utcfromtimestamp(start_ts+working_unit).strftime('%Y-%m-%d'))

        elif total_count < 200:
            working_unit = month_unit
            c_date = str(datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d'))+'..'+str(datetime.utcfromtimestamp(start_ts+working_unit).strftime('%Y-%m-%d'))
            request = 'https://api.github.com/search/repositories?q=created:"%s"language:python&sort=stars&order=desc&per_page=100' % c_date
            response =requests.get(request, headers=headers, auth=auth)
            
            total_count = response.json()['total_count']
            n_pages = min(int(total_count/100)+1, 10)
            for page in range(n_pages):
                request = 'https://api.github.com/search/repositories?q=created:"%s"language:python&sort=stars&order=desc&per_page=100&page=%d' % (c_date, page)
                response =requests.get(request, headers=headers, auth=auth)
                if len(concerned_projects)+total_count > goal:
                    to_download = goal - len(concerned_projects)
                else:
                    to_download = 100
                concerned_projects += download_projects(response, to_download, base_dir=base_dir)
            
            working_unit = week_unit
            start_ts = start_ts + month_unit
            c_date = str(datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d'))+'..'+str(datetime.utcfromtimestamp(start_ts+working_unit).strftime('%Y-%m-%d'))

        else:
            c_date = str(datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d'))+'..'+str(datetime.utcfromtimestamp(start_ts+working_unit).strftime('%Y-%m-%d'))
            request = 'https://api.github.com/search/repositories?q=created:"%s"language:python&sort=stars&order=desc&per_page=100' % c_date
            response =requests.get(request, headers=headers, auth=auth)
            total_count = response.json()['total_count']
            n_pages = min(int(total_count/100)+1, 10)

            for page in range(n_pages):
                request = 'https://api.github.com/search/repositories?q=created:"%s"language:python&sort=stars&order=desc&per_page=100&page=%d' % (c_date, page)
                response =requests.get(request, headers=headers, auth=auth)
                if len(concerned_projects)+total_count > goal:
                    to_download = goal - len(concerned_projects)
                else:
                    to_download = 100
                concerned_projects += download_projects(response, to_download, base_dir=base_dir)
            
            working_unit = week_unit
            start_ts = start_ts + week_unit
            c_date = str(datetime.utcfromtimestamp(start_ts).strftime('%Y-%m-%d'))+'..'+str(datetime.utcfromtimestamp(start_ts+working_unit).strftime('%Y-%m-%d'))
    return concerned_projects

if __name__ == '__main__':
    print("Inside main of scrape_repos")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
chromedriver_path = "/workspace/chromedriver"
service = ChromeService(executable_path=chromedriver_path)
d = webdriver.Chrome(service=service, options=chrome_options)
d.get("https://www.google.nl/")

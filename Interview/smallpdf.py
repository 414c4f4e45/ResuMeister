import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class smallpdf:
    """
    A class to interact with SmallPDF's AI PDF analysis tool to extract skills and experience from a PDF file.

    Attributes:
        file (str): The path to the PDF file to be analyzed.
        url (str): The URL of the SmallPDF AI PDF analysis page.
        options (Options): Selenium options for the Chrome driver.
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        skills (dict): Dictionary to store extracted skills and their descriptions.
        experience (dict): Dictionary to store extracted experience details and their descriptions.
        start (float): The start time for measuring execution duration.
    """
    def __init__(self, file=""):
        """
        Initializes the smallpdf class with the path to the PDF file and sets up the Selenium WebDriver.

        Args:
            file (str): The path to the PDF file to be analyzed.
        """
        self.url = f"https://smallpdf.com/ai-pdf"
        self.options = Options()
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--start-maximized")
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=self.options)
        self.file = os.path.abspath(file)
        self.skills = {}
        self.experience = {}
        self.start = time.time()
        self.time = 0
    def run(self):
        """
        Navigates to the SmallPDF AI PDF analysis page, performs skill and experience extraction, 
        and returns the execution duration.

        Returns:
            float: The time taken to complete the extraction process.
        """
        self.driver.get(self.url)
        self.extract_skills()
        self.extract_experience()
        self.driver.quit()
        self.time = time.time() - self.start

    def extract_skills(self):
        """
        Extracts skills from the PDF by uploading the file, submitting a query for skills, 
        and parsing the response to populate the skills attribute.

        This method assumes that the AI's response includes a summary of skills, 
        which is then parsed and organized into the skills dictionary.
        """
        file_input = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//input[@type='file']")))
        file_input.send_keys(self.file)
        
        summary_div = WebDriverWait(self.driver, 60).until(EC.presence_of_element_located((By.XPATH, "//div[@type='summary']")))
        summary_elements = self.driver.find_elements(By.XPATH, "//div[@type='summary']//p")
        summary_texts = [element.text for element in summary_elements]
        
        div_input = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//input[@type='text' and @placeholder='Hey! Ask me anything about your PDF.']")))
        div_input.send_keys("""analyze the pdf and extract all relevant keywords or skills mentioned.Focus on terms that represent specific skills, tools, technologies, or competencies. List these keywords separately and organize them if applicable (e.g., grouping programming languages, tools, or soft skills). But dont include anything from experience or internship section.""")
        
        div_button_submit = self.driver.find_elements(By.XPATH, "//button[@type='submit']")[2]
        div_button_submit.click()
        
        answer_div = WebDriverWait(self.driver, 60).until(EC.presence_of_element_located((By.XPATH, "//div[@type='answer']")))
        
        headings = self.driver.find_elements(By.XPATH, "//div[@type='answer']//h1")[1:]
        for heading in headings:
            title = heading.text
            list_items = heading.find_elements(By.XPATH, 'following-sibling::ul[1]//p')
            paragraphs = [item.text for item in list_items]
            self.skills[title] = paragraphs

    def extract_experience(self):
        """
        Extracts experience and training details from the PDF by submitting a query and 
        parsing the AI's response to populate the experience attribute.

        This method assumes that the AI's response includes details about training, industry, 
        and experience, which are then parsed and organized into the experience dictionary.
        """
        div_input = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, "//input[@type='text' and @placeholder='Hey! Ask me anything about your PDF.']")))
        div_input.send_keys("Now give me information from the training, Industry, experience section dont include a heading only subheadings are allowed")

        div_button_submit = self.driver.find_elements(By.XPATH, "//button[@type='submit']")[2]
        div_button_submit.click()
        
        WebDriverWait(self.driver, 60).until(lambda driver: len(driver.find_elements(By.XPATH, "//div[@type='answer']")) > 1)
        training_exp = self.driver.find_elements(By.XPATH, "//div[@type='answer']")[1]
        
        headings = training_exp.find_elements(By.XPATH, ".//h1")
        for heading in headings:
            title = heading.text
            list_items = heading.find_elements(By.XPATH, 'following-sibling::ul[1]//p')
            paragraphs = [item.text for item in list_items]
            self.experience[title] = paragraphs


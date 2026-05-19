from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from selenium.common.exceptions import NoSuchElementException
import html2text

def html_to_markdown(html_content):
    """Convert HTML to Markdown format"""
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = False
    converter.ignore_tables = False
    converter.body_width = 0  # Don't wrap text
    
    return converter.handle(html_content)

def get_transcript(ticker):
    url = f"https://www.roic.ai/quote/{ticker}/transcripts"
    
    # Set up stealthy Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Initialize driver
    driver = webdriver.Chrome(options=options)
    
    # Apply stealth settings
    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )
    
    try:
        driver.get(url)
        
        # Try to find the transcript element, catch if not found
        try:
            element = driver.find_element(By.CSS_SELECTOR, 
                ".flex.flex-col.rounded-lg.border.border-border-accent.text-foreground.shadow")
            
            # Get HTML from the element
            html = element.get_attribute('innerHTML')
            print(f"Found transcript element ({len(html)} characters)")
            
            return html
            
        except NoSuchElementException:
            print(f"This ticker has no transcripts available in the URL: {url}")
            return None
    
    finally:
        driver.quit()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from selenium.common.exceptions import NoSuchElementException
import html2text
import os

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

if __name__ == "__main__":
    ticker = "PYPL"
    print(f"Getting transcript for {ticker}...")
    
    try:
        html = get_transcript(ticker)
        
        if html:
            # Create data/transcripts directory if it doesn't exist
            os.makedirs("data/transcripts", exist_ok=True)
            
            # Save HTML version
            html_path = f"data/transcripts/{ticker}_transcript.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Saved HTML to {html_path}")
            
            # Convert to Markdown and save
            markdown = html_to_markdown(html)
            md_path = f"data/transcripts/{ticker}_transcript.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"Saved Markdown to {md_path}")
        else:
            print("No transcript data to save")
        
    except Exception as e:
        print(f"Error: {e}") 
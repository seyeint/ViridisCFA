from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from selenium.common.exceptions import TimeoutException
import html2text

def html_to_markdown(html_content):
    """Convert HTML to Markdown format"""
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = False
    converter.ignore_tables = False
    converter.body_width = 0  # Don't wrap text
    
    return converter.handle(html_content)

def _ticker_candidates(ticker):
    """Return source-specific ticker variants for transcript lookup."""
    clean = ticker.strip().upper()
    candidates = [clean]
    hyphenated = clean.replace(".", "-")
    if hyphenated not in candidates:
        candidates.append(hyphenated)
    return candidates

def get_transcript(ticker):
    ticker_candidates = _ticker_candidates(ticker)
    
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
        for candidate in ticker_candidates:
            url = f"https://www.roic.ai/quote/{candidate}/transcripts"
            driver.get(url)
            
            # Try to find the transcript element, catch if not found
            try:
                element = WebDriverWait(driver, 12).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        ".flex.flex-col.rounded-lg.border.border-border-accent.text-foreground.shadow"
                    ))
                )

                # Get HTML from the element
                html = element.get_attribute('innerHTML')
                if candidate != ticker.strip().upper():
                    print(f"Found transcript element using ticker fallback {candidate} ({len(html)} characters)")
                else:
                    print(f"Found transcript element ({len(html)} characters)")

                return html

            except TimeoutException:
                print(f"This ticker has no transcripts available in the URL: {url}")
        return None

    finally:
        driver.quit()

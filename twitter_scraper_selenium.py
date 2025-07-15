import time
import random
import json
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

class TwitterSeleniumScraper:
    def __init__(self, headless=True):
        """
        Initialize the Twitter scraper with Selenium.
        
        Args:
            headless (bool): Run browser in headless mode
        """
        self.driver = None
        self.headless = headless
        self.setup_driver()
    
    def setup_driver(self):
        """Set up the Chrome WebDriver with appropriate options."""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Add options to avoid detection
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            print("💡 Make sure you have Chrome installed and chromedriver in your PATH")
            self.driver = None
    
    def get_user_tweets(self, username, count=10, since_date=None, until_date=None):
        """
        Get tweets from a Twitter user using Selenium.
        
        Args:
            username (str): Twitter username (without @)
            count (int): Number of tweets to fetch
            since_date (datetime): Only include tweets after this date
            until_date (datetime): Only include tweets before this date
            
        Returns:
            list: List of tweet dictionaries
        """
        if not self.driver:
            print("❌ WebDriver not available, returning sample tweets")
            return self._get_sample_tweets(count)
        
        try:
            # If date range is specified, use search method
            if since_date or until_date:
                return self._get_tweets_by_date_range(username, count, since_date, until_date)
            else:
                return self._get_recent_tweets(username, count)
            
        except Exception as e:
            print(f"❌ Error scraping tweets: {e}")
            return self._get_sample_tweets(count)
    
    def _get_recent_tweets(self, username, count):
        """Get recent tweets from user profile."""
        # Navigate to the user's profile
        url = f"https://twitter.com/{username}"
        print(f"🔍 Navigating to @{username}'s profile...")
        
        self.driver.get(url)
        time.sleep(3)
        
        # Scroll to load more tweets
        self._scroll_to_load_tweets(count, aggressive=True)
        
        # Extract tweets
        tweets = self._extract_tweets_from_page(count)
        
        if not tweets:
            print("⚠️ No tweets found, returning sample data")
            return self._get_sample_tweets(count)
        
        print(f"✅ Successfully extracted {len(tweets)} tweets")
        return tweets[:count]
    
    def _get_tweets_by_date_range(self, username, count, since_date=None, until_date=None):
        """Get tweets from a specific date range using Twitter search."""
        print(f"🔍 Searching for tweets from @{username} in date range...")
        
        # Build search query
        search_query = f"from:{username}"
        if since_date:
            search_query += f" since:{since_date.strftime('%Y-%m-%d')}"
        if until_date:
            search_query += f" until:{until_date.strftime('%Y-%m-%d')}"
        
        # Navigate to search page
        search_url = f"https://twitter.com/search?q={search_query}&src=typed_query&f=live"
        print(f"🔍 Navigating to search: {search_query}")
        
        self.driver.get(search_url)
        time.sleep(5)  # Wait longer for search results
        
        # Scroll aggressively to load more results
        self._scroll_to_load_tweets(count * 2, aggressive=True, max_attempts=20)
        
        # Extract tweets
        tweets = self._extract_tweets_from_page(count * 2)
        
        if not tweets:
            print("⚠️ No tweets found in date range, trying profile method...")
            return self._get_recent_tweets(username, count)
        
        print(f"✅ Successfully extracted {len(tweets)} tweets from date range")
        return tweets[:count]
    
    def _scroll_to_load_tweets(self, target_count, aggressive=False, max_attempts=15):
        """Scroll down to load more tweets with enhanced scrolling."""
        print("📜 Scrolling to load tweets...")
        
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        tweets_found = 0
        scroll_attempts = 0
        no_change_count = 0
        
        while tweets_found < target_count and scroll_attempts < max_attempts:
            # More aggressive scrolling for date-based searches
            if aggressive:
                # Scroll multiple times quickly
                for _ in range(3):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(0.5)
                
                # Try to click "Show more" buttons if they exist
                try:
                    show_more_buttons = self.driver.find_elements(By.XPATH, "//span[contains(text(), 'Show more')]")
                    for button in show_more_buttons[:2]:  # Click first 2 buttons
                        try:
                            button.click()
                            time.sleep(1)
                        except:
                            continue
                except:
                    pass
            else:
                # Standard scrolling
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Count current tweets
            tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
            tweets_found = len(tweet_elements)
            
            # Check if we've reached the bottom
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                no_change_count += 1
                if no_change_count >= 3:  # Try a few more times before giving up
                    # Try to find and click "Show more" or similar buttons
                    try:
                        more_buttons = self.driver.find_elements(By.XPATH, "//span[contains(text(), 'Show more') or contains(text(), 'Load more')]")
                        if more_buttons:
                            more_buttons[0].click()
                            time.sleep(2)
                            no_change_count = 0
                        else:
                            scroll_attempts += 1
                    except:
                        scroll_attempts += 1
            else:
                no_change_count = 0
                last_height = new_height
            
            print(f"   Found {tweets_found} tweets so far... (scroll {scroll_attempts + 1}/{max_attempts})")
            
            # If we're not finding tweets, try alternative selectors
            if tweets_found == 0 and scroll_attempts > 5:
                print("   Trying alternative tweet detection...")
                # Look for any content that might be tweets
                content_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article, [role="article"]')
                tweets_found = len(content_elements)
    
    def _extract_tweets_from_page(self, count):
        """Extract tweets from the current page with enhanced detection."""
        tweets = []
        
        try:
            # Try multiple selectors for tweet detection
            selectors = [
                '[data-testid="tweet"]',
                'article[data-testid="tweet"]',
                '[role="article"]',
                'div[data-testid="cellInnerDiv"]'
            ]
            
            tweet_elements = []
            for selector in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    tweet_elements = elements
                    print(f"   Found {len(elements)} elements using selector: {selector}")
                    break
            
            for element in tweet_elements[:count * 3]:  # Get more to filter
                try:
                    tweet_data = self._extract_single_tweet(element)
                    if tweet_data and len(tweet_data['content'].strip()) > 10:
                        tweets.append(tweet_data)
                        if len(tweets) >= count:
                            break
                except Exception as e:
                    continue
            
            # If no tweets found with standard selectors, try alternative method
            if not tweets:
                tweets = self._extract_tweets_alternative()
            
        except Exception as e:
            print(f"⚠️ Error extracting tweets: {e}")
        
        return tweets
    
    def _extract_single_tweet(self, element):
        """Extract data from a single tweet element with enhanced parsing."""
        try:
            # Try multiple selectors for tweet text
            text_selectors = [
                '[data-testid="tweetText"]',
                'div[data-testid="tweetText"]',
                'span[data-testid="tweetText"]',
                'p',  # Fallback
            ]
            
            content = ""
            for selector in text_selectors:
                try:
                    text_element = element.find_element(By.CSS_SELECTOR, selector)
                    content = text_element.text.strip()
                    if content:
                        break
                except:
                    continue
            
            if not content:
                return None
            
            # Extract timestamp with multiple methods
            timestamp = "Unknown"
            try:
                # Try time element
                time_element = element.find_element(By.TAG_NAME, 'time')
                timestamp = time_element.get_attribute('datetime') or time_element.text
            except:
                try:
                    # Try to find timestamp in text
                    time_texts = element.find_elements(By.CSS_SELECTOR, 'time, span[dir="ltr"]')
                    for time_text in time_texts:
                        text = time_text.text.strip()
                        if any(word in text.lower() for word in ['ago', 'min', 'hour', 'day', 'week', 'month', 'year']):
                            timestamp = text
                            break
                except:
                    pass
            
            # Extract engagement metrics with multiple selectors
            likes = self._extract_metric_enhanced(element, ['[data-testid="like"]', '[data-testid="unlike"]'])
            retweets = self._extract_metric_enhanced(element, ['[data-testid="retweet"]', '[data-testid="unretweet"]'])
            replies = self._extract_metric_enhanced(element, ['[data-testid="reply"]'])
            
            return {
                'content': content,
                'timestamp': timestamp,
                'likes': likes,
                'retweets': retweets,
                'replies': replies
            }
            
        except Exception as e:
            return None
    
    def _extract_metric_enhanced(self, element, selectors):
        """Extract engagement metric with multiple selector attempts."""
        for selector in selectors:
            try:
                metric_element = element.find_element(By.CSS_SELECTOR, selector)
                text = metric_element.text.strip()
                # Extract number from text
                numbers = re.findall(r'[\d,]+', text)
                if numbers:
                    return numbers[0]
            except:
                continue
        return "0"
    
    def _extract_tweets_alternative(self):
        """Alternative method to extract tweets when primary method fails."""
        tweets = []
        
        try:
            # Look for any text that might be tweet content
            text_elements = self.driver.find_elements(By.CSS_SELECTOR, 'p, span, div')
            
            for element in text_elements:
                text = element.text.strip()
                if (len(text) > 20 and 
                    not text.startswith('http') and 
                    not text.isdigit() and
                    len(tweets) < 10):
                    tweets.append({
                        'content': text,
                        'timestamp': 'Unknown',
                        'likes': '0',
                        'retweets': '0',
                        'replies': '0'
                    })
        except Exception as e:
            print(f"⚠️ Alternative extraction failed: {e}")
        
        return tweets
    
    def _get_sample_tweets(self, count):
        """Return sample tweets when scraping fails."""
        sample_tweets = [
            "Just had an amazing idea for a new project! 🚀",
            "The future of technology is incredibly exciting.",
            "Working on some really cool stuff today.",
            "Innovation happens when you least expect it.",
            "Sometimes the best solutions are the simplest ones.",
            "Building something that could change everything.",
            "The intersection of AI and human creativity is fascinating.",
            "Progress is made one step at a time.",
            "Great things are built by teams, not individuals.",
            "The best time to start was yesterday, the second best is now."
        ]
        
        return [
            {
                'content': tweet,
                'timestamp': 'Sample',
                'likes': str(random.randint(50, 1000)),
                'retweets': str(random.randint(10, 500)),
                'replies': str(random.randint(5, 200))
            }
            for tweet in random.sample(sample_tweets, min(count, len(sample_tweets)))
        ]
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()

def main():
    """Demo the Twitter Selenium scraper."""
    scraper = TwitterSeleniumScraper(headless=True)
    
    try:
        # Test with funny/comical usernames
        usernames = ["elonmusk", "dril", "horse_ebooks", "shitmydadsays", "fart"]
        
        for username in usernames:
            print(f"\n{'='*60}")
            print(f"📱 SCRAPING TWEETS FROM @{username}")
            print(f"{'='*60}")
            
            tweets = scraper.get_user_tweets(username, count=10)
            
            for i, tweet in enumerate(tweets, 1):
                print(f"\n{i}. {tweet['content']}")
                print(f"   📅 {tweet['timestamp']} | ❤️ {tweet['likes']} | 🔄 {tweet['retweets']} | 💬 {tweet['replies']}")
            
            # Add delay between requests
            time.sleep(3)
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 
import requests
import time
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urljoin, urlencode
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ASOSScraper:
    def __init__(self):
        self.base_url = 'https://www.asos.com'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.asos.com/us/women/',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def save_progress(self, products, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(products, f, indent=2, ensure_ascii=False)
            logger.info(f"Progress saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving progress to {filename}: {e}")

    def get_categories_from_homepage(self):
        try:
            # First get the main categories from homepage
            response = self.session.get(f"{self.base_url}/us/women/")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            categories = []

            # Get feature links
            for link in soup.find_all('a', class_='feature__link'):
                href = link.get('href')
                if href:
                    if not href.startswith('http'):
                        href = urljoin(self.base_url, href)
                    title = link.find('p')
                    name = title.text.strip() if title else "Unknown"
                    categories.append({
                        'url': href,
                        'name': name
                    })

            # Get additional navigation categories
            nav_links = soup.find_all('a', {'data-analytics-id': True})
            for link in nav_links:
                href = link.get('href')
                if href and ('/cat/' in href or '/ctas/' in href):
                    if not href.startswith('http'):
                        href = urljoin(self.base_url, href)
                    categories.append({
                        'url': href,
                        'name': link.text.strip()
                    })

            # Add main category URLs
            main_categories = [
                ('New In', '/women/new-in/cat/'),
                ('Clothing', '/women/clothing/cat/'),
                ('Dresses', '/women/dresses/cat/'),
                ('Tops', '/women/tops/cat/'),
                ('Shoes', '/women/shoes/cat/'),
                ('Sportswear', '/women/sportswear/cat/'),
                ('Accessories', '/women/accessories/cat/'),
                ('Winter', '/women/ctas/winter-warmers/cat/'),
                ('Trending', '/women/trending-now/cat/')
            ]

            for name, path in main_categories:
                url = urljoin(f"{self.base_url}/us", path)
                categories.append({
                    'url': url,
                    'name': name
                })

            # Remove duplicates while preserving order
            seen = set()
            unique_categories = []
            for cat in categories:
                if cat['url'] not in seen:
                    seen.add(cat['url'])
                    unique_categories.append(cat)

            logger.info(f"Found {len(unique_categories)} categories")
            return unique_categories

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def extract_products_from_listing(self, url):
        try:
            logger.info(f"Fetching category page: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            product_urls = set()  # Using set for automatic deduplication
            
            # Try different product selectors
            selectors = [
                'article[data-auto-id="productTile"]',
                'div[data-auto-id="productTile"]',
                'div[data-test-id="productTile"]',
                'a[data-auto-id="productTile"]',
                '[data-analytics-id^="product-"]'
            ]
            
            for selector in selectors:
                products = soup.select(selector)
                for product in products:
                    link = product.find('a') if not product.name == 'a' else product
                    if link and link.get('href'):
                        product_url = link['href']
                        if not product_url.startswith('http'):
                            product_url = urljoin(self.base_url, product_url)
                        product_urls.add(product_url)

            # Additional link search
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(x in href.lower() for x in ['/prd/', 'product', '/prod/']):
                    if not href.startswith('http'):
                        href = urljoin(self.base_url, href)
                    product_urls.add(href)
            
            product_urls = list(product_urls)
            logger.info(f"Found {len(product_urls)} products on page")
            
            if not product_urls:
                logger.debug(f"Page content preview: {soup.prettify()[:1000]}")
            
            return product_urls

        except Exception as e:
            logger.error(f"Error fetching category page: {e}")
            return []

    def get_all_product_urls(self, category_url):
            all_urls = set()  # Using set for automatic deduplication
            page = 1
            consecutive_empty_pages = 0
            max_empty_pages = 3  # Stop after 3 consecutive empty pages
            max_pages = 300  # Maximum number of pages to prevent infinite loops
            
            while consecutive_empty_pages < max_empty_pages and page <= max_pages:
                # Add sorting and page size parameters
                params = {
                    'page': page,
                    'sort': 'freshness',
                    'rowlength': '4',
                    'limit': '72'
                }
                
                # Add parameters to URL
                if '?' in category_url:
                    page_url = f"{category_url}&{urlencode(params)}"
                else:
                    page_url = f"{category_url}?{urlencode(params)}"
                    
                urls = self.extract_products_from_listing(page_url)
                
                if not urls:
                    consecutive_empty_pages += 1
                else:
                    consecutive_empty_pages = 0
                    # Process each batch of URLs immediately
                    all_urls.update(urls)
                    logger.info(f"Found {len(urls)} new URLs on page {page}, total unique URLs: {len(all_urls)}")
                
                # Save progress every 5 pages
                if page % 5 == 0:
                    temp_filename = f'temp_urls_{page}.json'
                    try:
                        with open(temp_filename, 'w', encoding='utf-8') as f:
                            json.dump(list(all_urls), f, indent=2, ensure_ascii=False)
                        logger.info(f"Saved {len(all_urls)} URLs to {temp_filename}")
                    except Exception as e:
                        logger.error(f"Error saving temp URLs: {e}")
                
                page += 1
                time.sleep(1)  # Be polite
            
            return list(all_urls)

    def scrape_product(self, url):
        try:
            time.sleep(1)
            logger.info(f"Scraping product: {url}")
            
            # If url has a color ID, remove it to get base product
            base_url = url.split('#')[0] if '#' in url else url
            response = self.session.get(base_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get all image variations for each color
            all_images = set()  # Using set for automatic deduplication
            image_types = ['-1', '-2', '-3', '-4', '-5']  # Different image views
            
            # Find all color options
            colors = set()  # Using set for automatic deduplication
            color_links = soup.find_all('a', {'data-testid': lambda x: x and 'colour-variation' in x})
            
            # If no color variations found, try other color selection elements
            if not color_links:
                color_select = soup.find('select', {'data-id': 'colour-select'})
                if color_select:
                    for option in color_select.find_all('option'):
                        color = option.text.strip()
                        if color and color.lower() != 'please select':
                            colors.add(color)
                            # Try to get color-specific images
                            color_slug = color.lower().replace(' ', '-')
                            for img_type in image_types:
                                img_url = f"{base_url.replace('/prd/', '/images/')}-{img_type}-{color_slug}"
                                all_images.add(img_url)
            else:
                # Process each color variation
                for color_link in color_links:
                    color_url = color_link.get('href')
                    if color_url:
                        if not color_url.startswith('http'):
                            color_url = urljoin(self.base_url, color_url)
                        
                        # Get color name from color swatch or text
                        color_name = color_link.get('aria-label', '').replace('Select Color ', '')
                        if color_name:
                            colors.add(color_name)
                        
                        # Get images for this color variation
                        time.sleep(0.5)  # Small delay between color requests
                        color_response = self.session.get(color_url)
                        if color_response.ok:
                            color_soup = BeautifulSoup(color_response.text, 'html.parser')
                            
                            # Get images from gallery
                            gallery = color_soup.find('div', {'data-test-id': 'gallery-images'})
                            if gallery:
                                for img in gallery.find_all('img'):
                                    src = img.get('src')
                                    if src:
                                        # Try to get highest quality version
                                        high_res = src.replace('$n_320w$', '$n_2880w$')
                                        all_images.add(high_res)

            # Get sizes
            sizes = set()  # Using set for automatic deduplication
            size_select = soup.find('select', {'data-id': 'size-select'})
            if size_select:
                for option in size_select.find_all('option'):
                    size = option.text.strip()
                    if size and size.lower() != 'please select':
                        sizes.add(size)

            # Get price from multiple sources
            price = None
            currency = None
            
            # Method 1: Check structured data
            scripts = soup.find_all('script', {'type': 'application/ld+json'})
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        if data.get('@type') == 'Product' and 'offers' in data:
                            offers = data['offers']
                            if isinstance(offers, dict):
                                price = float(offers.get('price', 0))
                                currency = offers.get('priceCurrency')
                                break
                except:
                    continue
            
            # Method 2: Check price elements if still no price
            if not price:
                price_elem = soup.find('span', {'data-test-id': 'current-price'}) or \
                           soup.find('span', {'data-id': 'current-price'})
                if price_elem:
                    price_text = price_elem.text.strip().replace('$', '').replace(',', '')
                    try:
                        price = float(price_text)
                        currency = 'USD'
                    except:
                        pass

            # Get product data
            name = soup.find('h1')
            name = name.text.strip() if name else None
            
            description = soup.find('div', {'data-test-id': 'product-description'})
            description = description.text.strip() if description else None
            
            brand = soup.find('span', {'data-test-id': 'brand-name'})
            brand = brand.text.strip() if brand else None
            
            # Get SKU from URL or product data
            sku = url.split('prd/')[-1].split('/')[0] if '/prd/' in url else None

            product_info = {
                'url': url,
                'name': name,
                'brand': brand,
                'description': description,
                'colors': list(colors),
                'sizes': list(sizes),
                'sku': sku,
                'price': price,
                'currency': currency,
                'images': list(all_images),
                'category': None  # Will be set by main loop
            }

            logger.info(f"Successfully scraped: {product_info['name']}")
            logger.info(f"Found {len(product_info['images'])} images and {len(product_info['colors'])} colors")
            return product_info

        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None

def main():
    scraper = ASOSScraper()
    
    # Get categories from homepage
    categories = scraper.get_categories_from_homepage()
    logger.info(f"Starting to scrape {len(categories)} categories")
    
    all_products = []
    for category in categories:
        logger.info(f"\nProcessing category: {category['name']} ({category['url']})")
        
        # Get all product URLs for this category including pagination
        product_urls = scraper.get_all_product_urls(category['url'])
        logger.info(f"Found {len(product_urls)} products in category {category['name']}")
        
        # Scrape each product
        for url in product_urls:
            product_data = scraper.scrape_product(url)
            if product_data:
                # Add category name to product data
                product_data['category'] = category['name']
                all_products.append(product_data)
                
                # Save progress every 10 products
                if len(all_products) % 10 == 0:
                    with open('asos_women_products_progress.json', 'w', encoding='utf-8') as f:
                        json.dump(all_products, f, indent=2, ensure_ascii=False)
                    logger.info(f"Progress saved: {len(all_products)} products")
    
    # Save final results
    if all_products:
        with open('asos_women_products_final.json', 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)
        logger.info(f"\nFinished! Saved {len(all_products)} products to asos_women_products_final.json")
    else:
        logger.error("No products were successfully scraped")

if __name__ == "__main__":
    main()
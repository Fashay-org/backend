import requests
import time
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZaraMenScraper:
    def __init__(self):
        self.base_url = 'https://www.zara.com/us'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_men_categories(self):
        try:
            categories = [
                '/en/man-outerwear-l715.html',            # JACKETS | OUTERWEAR
                '/en/man-blazers-l608.html',              # BLAZERS
                '/en/man-shirts-l737.html',               # SHIRTS
                '/en/man-t-shirts-l855.html',             # T-SHIRTS
                '/en/man-polo-shirts-l838.html',          # POLO SHIRTS
                '/en/man-sweaters-cardigans-l821.html',   # SWEATERS | CARDIGANS
                '/en/man-sweatshirts-l819.html',          # SWEATSHIRTS
                '/en/man-trousers-l838.html',             # TROUSERS
                '/en/man-jeans-l659.html',                # JEANS
                '/en/man-shorts-l902.html',               # SHORTS
                '/en/man-suits-l808.html',                # SUITS
                '/en/man-shoes-l769.html',                # SHOES
                '/en/man-bags-l563.html',                 # BAGS
                '/en/man-accessories-l537.html',          # ACCESSORIES
                '/en/man-swimwear-l807.html',             # SWIMWEAR
                '/en/man-perfumes-l711.html'              # FRAGRANCES
            ]
            
            full_urls = [f"{self.base_url}{cat}" for cat in categories]
            logger.info(f"Found {len(full_urls)} categories")
            return full_urls
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def extract_products_from_listing(self, url):
        try:
            logger.info(f"Fetching category page: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            product_urls = []
            
            # Try to get JSON-LD data first (contains listing info)
            json_ld = soup.find('script', {'type': 'application/ld+json'})
            if json_ld and json_ld.string:
                try:
                    data = json.loads(json_ld.string)
                    if data.get('@type') == 'ItemList' and data.get('itemListElement'):
                        for item in data['itemListElement']:
                            product = item.get('item', {})
                            if product.get('offers', {}).get('url'):
                                product_urls.append(product['offers']['url'])
                except json.JSONDecodeError:
                    pass

            # Fallback to HTML parsing if no JSON-LD data
            if not product_urls:
                for product in soup.find_all('a', class_='product-link'):
                    if product.get('href'):
                        product_urls.append(product['href'])
            
            logger.info(f"Found {len(product_urls)} products on page")
            return list(set(product_urls))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error fetching category page: {e}")
            return []

    def get_all_product_urls(self, category_url):
        all_urls = []
        page = 1
        while True:
            page_url = f"{category_url}?page={page}"
            urls = self.extract_products_from_listing(page_url)
            if not urls:
                break
            all_urls.extend(urls)
            page += 1
            time.sleep(1)
        return list(set(all_urls))

    def scrape_product_images(self, soup):
        images = set()  # Using set to avoid duplicates
        
        # Method 1: Get images from product grid
        for img in soup.find_all('img', class_='media-image__image'):
            src = img.get('src')
            if src and 'transparent-background' not in src:
                images.add(src)
                
            # Check srcset for higher quality images
            srcset = img.get('srcset')
            if srcset:
                # Get highest quality image from srcset
                srcset_urls = [url.strip().split(' ')[0] for url in srcset.split(',')]
                highest_quality = srcset_urls[-1] if srcset_urls else None
                if highest_quality and 'transparent-background' not in highest_quality:
                    images.add(highest_quality)
                    
        # Method 2: Get images from product detail view
        media_containers = soup.find_all('div', class_='media__wrapper')
        for container in media_containers:
            # Check for data-src attribute
            data_src = container.get('data-src')
            if data_src and 'transparent-background' not in data_src:
                images.add(data_src)
                
            # Check child img elements
            img = container.find('img')
            if img:
                src = img.get('src')
                if src and 'transparent-background' not in src:
                    images.add(src)
                    
                # Check srcset again
                srcset = img.get('srcset')
                if srcset:
                    srcset_urls = [url.strip().split(' ')[0] for url in srcset.split(',')]
                    highest_quality = srcset_urls[-1] if srcset_urls else None
                    if highest_quality and 'transparent-background' not in highest_quality:
                        images.add(highest_quality)
        
        # Method 3: Get images from product detail carousel
        carousel_items = soup.find_all('li', class_='products-category-grid-media-carousel-item')
        for item in carousel_items:
            img = item.find('img')
            if img:
                src = img.get('src')
                if src and 'transparent-background' not in src:
                    images.add(src)
                    
                srcset = img.get('srcset')
                if srcset:
                    srcset_urls = [url.strip().split(' ')[0] for url in srcset.split(',')]
                    highest_quality = srcset_urls[-1] if srcset_urls else None
                    if highest_quality and 'transparent-background' not in highest_quality:
                        images.add(highest_quality)

        # Remove any None values and return as list
        return list(filter(None, images))

    def scrape_product(self, url):
        try:
            time.sleep(1)
            logger.info(f"Scraping product: {url}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get JSON-LD data
            script_tags = soup.find_all('script', type='application/ld+json')
            product_data = None
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    # Check if it's product data (could be in list or single object form)
                    if isinstance(data, list):
                        product_data = data
                        break
                    elif isinstance(data, dict) and data.get('@type') == 'Product':
                        product_data = [data]
                        break
                except:
                    continue

            if not product_data:
                logger.warning(f"No valid product data found for {url}")
                return None

            # Use the first product for basic info
            base_product = product_data[0]
            
            # Get unique colors and sizes from all variations
            colors = []
            sizes = []
            images = []
            for variant in product_data:
                if 'color' in variant and variant['color'] not in colors:
                    colors.append(variant['color'])
                if 'size' in variant and variant['size'] not in sizes:
                    sizes.append(variant['size'])
                if 'image' in variant and variant['image'] not in images:
                    # Add both original and high-res versions
                    img_url = variant['image']
                    images.append(img_url)
                    # Try to get higher resolution version
                    high_res = img_url.replace('w=560', 'w=1920')
                    if high_res != img_url:
                        images.append(high_res)

            # Also check image carousels
            product_galleries = soup.find_all('source')
            for source in product_galleries:
                srcset = source.get('srcset', '')
                if srcset:
                    # Get all image URLs from srcset
                    srcs = [s.split(' ')[0] for s in srcset.split(',')]
                    # Get the highest resolution version
                    largest_img = max(srcs, key=lambda x: int(x.split('w=')[-1].split('&')[0]) if 'w=' in x else 0)
                    if largest_img and largest_img not in images:
                        images.append(largest_img)

            # Extract price safely
            price = None
            currency = None
            if 'offers' in base_product:
                offers = base_product['offers']
                if isinstance(offers, dict):
                    try:
                        price = float(offers.get('price', 0))
                        currency = offers.get('priceCurrency')
                    except (ValueError, TypeError):
                        pass

            product_info = {
                'url': url,
                'name': base_product.get('name'),
                'brand': base_product.get('brand'),
                'description': base_product.get('description'),
                'colors': colors,
                'sizes': sizes,
                'sku': base_product.get('sku'),
                'price': price,
                'currency': currency,
                'images': list(set(images)),  # Remove duplicates
                'category': None
            }
            
            # Get category
            breadcrumb = soup.find('nav', {'aria-label': 'Breadcrumb'})
            if breadcrumb:
                last_category = breadcrumb.find_all('a')[-1]
                if last_category:
                    product_info['category'] = last_category.text.strip()

            logger.info(f"Successfully scraped: {product_info['name']}")
            logger.info(f"Found {len(product_info['images'])} images for this product")
            return product_info

        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None

def main():
    scraper = ZaraMenScraper()
    
    # Get categories
    categories = scraper.get_men_categories()
    logger.info(f"Starting to scrape {len(categories)} categories")
    
    all_products = []
    for category_url in categories:
        logger.info(f"\nProcessing category: {category_url}")
        
        # Get all product URLs for this category including pagination
        product_urls = scraper.get_all_product_urls(category_url)
        
        # Scrape each product
        for url in product_urls:
            product_data = scraper.scrape_product(url)
            if product_data:
                all_products.append(product_data)
                
                # Save progress every 10 products
                if len(all_products) % 10 == 0:
                    with open('zara_men_products_progress.json', 'w', encoding='utf-8') as f:
                        json.dump(all_products, f, indent=2, ensure_ascii=False)
                    logger.info(f"Progress saved: {len(all_products)} products")
    
    # Save final results
    if all_products:
        with open('zara_men_products_final.json', 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)
        logger.info(f"\nFinished! Saved {len(all_products)} products to zara_men_products_final.json")
    else:
        logger.error("No products were successfully scraped")

if __name__ == "__main__":
    main()
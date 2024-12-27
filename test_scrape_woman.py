import requests
import time
from bs4 import BeautifulSoup
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HMScraper:
    def __init__(self):
        self.base_url = 'https://www2.hm.com'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_women_categories(self):
        try:
            url = 'https://www2.hm.com/en_us/women/products/view-all.html'
            response = self.session.get(url)
            response.raise_for_status()
            
            categories = [
                "https://www2.hm.com/en_us/women/products/cardigans-sweaters.html",
                "https://www2.hm.com/en_us/women/products/cardigans-sweaters/sweater-vests.html",
                "https://www2.hm.com/en_us/women/products/cardigans-sweaters/cardigans.html",
                "https://www2.hm.com/en_us/women/products/cardigans-sweaters/sweaters.html",
                "https://www2.hm.com/en_us/women/products/cardigans-sweaters/turtlenecks.html",
                "https://www2.hm.com/en_us/women/products/tops.html",
                "https://www2.hm.com/en_us/women/products/tops/low-cut.html",
                "https://www2.hm.com/en_us/women/products/tops/corsets.html",
                "https://www2.hm.com/en_us/women/products/tops/collared.html",
                "https://www2.hm.com/en_us/women/products/tops/turtleneck.html",
                "https://www2.hm.com/en_us/women/products/tops/halter.html",
                "https://www2.hm.com/en_us/women/products/tops/cut-out.html",
                "https://www2.hm.com/en_us/women/products/tops/bandeau.html",
                "https://www2.hm.com/en_us/women/products/tops/puff-sleeve.html",
                "https://www2.hm.com/en_us/women/products/tops/vest.html",
                "https://www2.hm.com/en_us/women/products/tops/short-sleeve.html",
                "https://www2.hm.com/en_us/women/products/tops/long-sleeve.html",
                "https://www2.hm.com/en_us/women/products/tops/crop.html",
                "https://www2.hm.com/en_us/women/products/tops/bodysuits.html",
                "https://www2.hm.com/en_us/women/products/tops/t-shirts.html",
                "https://www2.hm.com/en_us/women/products/tops/printed-graphic-t-shirts.html",
                "https://www2.hm.com/en_us/women/products/dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/one-shoulder.html",
                "https://www2.hm.com/en_us/women/products/dresses/backless.html",
                "https://www2.hm.com/en_us/women/products/dresses/cut-out.html",
                "https://www2.hm.com/en_us/women/products/dresses/sleeveless.html",
                "https://www2.hm.com/en_us/women/products/dresses/puffsleeve.html",
                "https://www2.hm.com/en_us/women/products/dresses/halterneck.html",
                "https://www2.hm.com/en_us/women/products/dresses/wedding-guest-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/a-line.html",
                "https://www2.hm.com/en_us/women/products/dresses/cami.html",
                "https://www2.hm.com/en_us/women/products/dresses/knitted-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/long-sleeve.html",
                "https://www2.hm.com/en_us/women/products/dresses/linen.html",
                "https://www2.hm.com/en_us/women/products/dresses/t-shirt-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/denim.html",
                "https://www2.hm.com/en_us/women/products/dresses/short-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/midi-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/maxi-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/beach.html",
                "https://www2.hm.com/en_us/women/products/dresses/bodycon-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/party-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/cocktail-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/lace-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/shirt-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/sequin-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/wrap-dresses.html",
                "https://www2.hm.com/en_us/women/products/dresses/kaftans.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/teddy.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/raincoats.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/anoraks.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/puffer-vests.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/puffer-vests/puffer.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/shirt-jackets.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/trench-coat.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/jackets.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/coats.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/bomber-jackets.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/winter-jackets.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/biker.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/denim-jackets.html",
                "https://www2.hm.com/en_us/women/products/jackets-coats/puffer.html",
                "https://www2.hm.com/en_us/women/products/jeans.html",
                "https://www2.hm.com/en_us/women/products/jeans/mom.html",
                "https://www2.hm.com/en_us/women/products/jeans/skinny.html",
                "https://www2.hm.com/en_us/women/products/jeans/wide-leg.html",
                "https://www2.hm.com/en_us/women/products/jeans/straight.html",
                "https://www2.hm.com/en_us/women/products/jeans/loose.html",
                "https://www2.hm.com/en_us/women/products/jeans/curvy-fit.html",
                "https://www2.hm.com/en_us/women/products/pants.html",
                "https://www2.hm.com/en_us/women/products/pants/linen.html",
                "https://www2.hm.com/en_us/women/products/pants/high-waisted.html",
                "https://www2.hm.com/en_us/women/products/pants/wide-leg.html",
                "https://www2.hm.com/en_us/women/products/pants/cargo.html",
                "https://www2.hm.com/en_us/women/products/pants/leggings.html",
                "https://www2.hm.com/en_us/women/products/pants/joggers.html",
                "https://www2.hm.com/en_us/women/products/pants/flare.html",
                "https://www2.hm.com/en_us/women/products/pants/chinos-slacks.html",
                "https://www2.hm.com/en_us/women/products/pants/dress-pants.html",
                "https://www2.hm.com/en_us/women/products/blazers-vests.html",
                "https://www2.hm.com/en_us/women/products/blazers-vests/oversized.html",
                "https://www2.hm.com/en_us/women/products/blazers-vests/fitted.html",
                "https://www2.hm.com/en_us/women/products/blazers-vests/double-breasted.html",
                "https://www2.hm.com/en_us/women/products/hoodies-sweatshirts.html",
                "https://www2.hm.com/en_us/women/products/hoodies-sweatshirts/hoodies.html",
                "https://www2.hm.com/en_us/women/products/hoodies-sweatshirts/sweatshirts.html",
                "https://www2.hm.com/en_us/women/products/shirts-blouses.html",
                "https://www2.hm.com/en_us/women/products/shirts-blouses/linen.html",
                "https://www2.hm.com/en_us/women/products/shirts-blouses/shirts.html",
                "https://www2.hm.com/en_us/women/products/shirts-blouses/blouses.html",
                "https://www2.hm.com/en_us/women/products/shirts-blouses/off-shoulder.html",
                "https://www2.hm.com/en_us/women/products/shirts-blouses/denim-shirts.html",
                "https://www2.hm.com/en_us/women/products/basics.html",
                "https://www2.hm.com/en_us/women/products/basics/tops.html",
                "https://www2.hm.com/en_us/women/products/basics/tops/vest.html",
                "https://www2.hm.com/en_us/women/products/basics/tops/short-sleeve.html",
                "https://www2.hm.com/en_us/women/products/basics/tops/long-sleeve.html",
                "https://www2.hm.com/en_us/women/products/basics/cardigans-sweaters.html",
                "https://www2.hm.com/en_us/women/products/basics/dresses-skirts.html",
                "https://www2.hm.com/en_us/women/products/basics/pants-leggings.html",
                "https://www2.hm.com/en_us/women/products/skirts.html",
                "https://www2.hm.com/en_us/women/products/skirts/mini.html",
                "https://www2.hm.com/en_us/women/products/skirts/pleated.html",
                "https://www2.hm.com/en_us/women/products/skirts/short-skirts.html",
                "https://www2.hm.com/en_us/women/products/skirts/midi-skirts.html",
                "https://www2.hm.com/en_us/women/products/skirts/pencil-skirts.html",
                "https://www2.hm.com/en_us/women/products/skirts/denim-skirts.html",
                "https://www2.hm.com/en_us/women/products/skirts/high-waisted-skirts.html",
                "https://www2.hm.com/en_us/women/products/skirts/skater-skirts.html",
                "https://www2.hm.com/en_us/women/products/skirts/wrap-skirts.html",
                "https://www2.hm.com/en_us/women/products/merch-graphics.html",
                "https://www2.hm.com/en_us/women/products/loungewear.html",
                "https://www2.hm.com/en_us/women/products/sleepwear.html",
                "https://www2.hm.com/en_us/women/products/sleepwear/nightgowns.html",
                "https://www2.hm.com/en_us/women/products/sleepwear/bathrobes-housecoats.html",
                "https://www2.hm.com/en_us/women/products/sleepwear/pajamas.html",
                "https://www2.hm.com/en_us/women/products/socks-tights.html",
                "https://www2.hm.com/en_us/women/products/socks-tights/leggings.html",
                "https://www2.hm.com/en_us/women/products/socks-tights/shaping.html",
                "https://www2.hm.com/en_us/women/products/jumpsuits-rompers.html",
                "https://www2.hm.com/en_us/women/products/jumpsuits-rompers/long-jumpsuits.html",
                "https://www2.hm.com/en_us/women/products/jumpsuits-rompers/playsuits-rompers.html",
                "https://www2.hm.com/en_us/women/products/h-m-edition.html",
                "https://www2.hm.com/en_us/women/products/premium-selection.html",
                "https://www2.hm.com/en_us/women/products/premium-selection/tops.html",
                "https://www2.hm.com/en_us/women/products/premium-selection/shoes.html",
                "https://www2.hm.com/en_us/women/products/premium-selection/jackets-coats.html",
                "https://www2.hm.com/en_us/women/products/premium-selection/dresses.html",
                "https://www2.hm.com/en_us/women/products/premium-selection/cardigans-sweaters.html",
                "https://www2.hm.com/en_us/women/products/premium-selection/pants.html",
                "https://www2.hm.com/en_us/women/products/premium-selection/accessories.html",
                "https://www2.hm.com/en_us/women/products/shorts.html",
                "https://www2.hm.com/en_us/women/products/shorts/biker.html",
                "https://www2.hm.com/en_us/women/products/shorts/bermudas.html",
                "https://www2.hm.com/en_us/women/products/shorts/denim-shorts.html",
                "https://www2.hm.com/en_us/women/products/shorts/high-waisted-shorts.html",
                "https://www2.hm.com/en_us/women/products/swimwear.html",
                "https://www2.hm.com/en_us/women/products/swimwear/bikini-sets.html",
                "https://www2.hm.com/en_us/women/products/swimwear/bikini-sets/bikini-tops.html",
                "https://www2.hm.com/en_us/women/products/swimwear/bikini-sets/bikini-bottoms.html",
                "https://www2.hm.com/en_us/women/products/swimwear/swimsuits.html",
                "https://www2.hm.com/en_us/women/products/swimwear/beachwear.html",
                "https://www2.hm.com/en_us/women/accessories/view-all.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/shoppers-tote-bags.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/crossbody-bags.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/shoulder-bags.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/phone-bags.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/backpacks.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/beach-bags.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/handbags.html",
                "https://www2.hm.com/en_us/women/accessories/purses-bags/gym-bags.html",
                "https://www2.hm.com/en_us/women/accessories/jewelry.html",
                "https://www2.hm.com/en_us/women/accessories/jewelry/bracelets.html",
                "https://www2.hm.com/en_us/women/accessories/jewelry/earrings.html",
                "https://www2.hm.com/en_us/women/accessories/jewelry/necklaces.html",
                "https://www2.hm.com/en_us/women/accessories/jewelry/rings.html",
                "https://www2.hm.com/en_us/women/accessories/sunglasses.html",
                "https://www2.hm.com/en_us/women/accessories/hair-accessories.html",
                "https://www2.hm.com/en_us/women/accessories/belts.html",
                "https://www2.hm.com/en_us/women/accessories/gloves.html",
                "https://www2.hm.com/en_us/women/accessories/hats.html",
                "https://www2.hm.com/en_us/women/accessories/hats/sun-hats.html",
                "https://www2.hm.com/en_us/women/accessories/hats/fedora-hats.html",
                "https://www2.hm.com/en_us/women/accessories/hats/bucket-hats.html",
                "https://www2.hm.com/en_us/women/accessories/scarves.html",
                "https://www2.hm.com/en_us/women/accessories/mobile-accessories.html",
                "https://www2.hm.com/en_us/women/shoes/view-all.html",
                "https://www2.hm.com/en_us/women/shoes/sneakers.html",
                "https://www2.hm.com/en_us/women/shoes/ballerinas.html",
                "https://www2.hm.com/en_us/women/shoes/boots.html",
                "https://www2.hm.com/en_us/women/shoes/espadrilles.html",
                "https://www2.hm.com/en_us/women/shoes/heels.html",
                "https://www2.hm.com/en_us/women/shoes/loafers.html",
                "https://www2.hm.com/en_us/women/shoes/mules.html",
                "https://www2.hm.com/en_us/women/shoes/slippers.html",
                "https://www2.hm.com/en_us/women/shoes/slingback.html",
                "https://www2.hm.com/en_us/women/sport/view-all.html",
                "https://www2.hm.com/en_us/women/sport/new-in.html",
                "https://www2.hm.com/en_us/women/sport/winter-destination.html",
                "https://www2.hm.com/en_us/women/sport/softmove.html",
                "https://www2.hm.com/en_us/women/sport/leggings-tights.html",
                "https://www2.hm.com/en_us/women/sport/leggings-tights/running.html",
                "https://www2.hm.com/en_us/women/sport/leggings-tights/seamless.html",
                "https://www2.hm.com/en_us/women/sport/leggings-tights/shaping.html",
                "https://www2.hm.com/en_us/women/sport/leggings-tights/yoga.html",
                "https://www2.hm.com/en_us/women/sport/leggings-tights/gym.html",
                "https://www2.hm.com/en_us/women/sport/sport-bras.html",
                "https://www2.hm.com/en_us/women/sport/tops.html",
                "https://www2.hm.com/en_us/women/sport/hoodies-sweatshirts.html",
                "https://www2.hm.com/en_us/women/sport/pants-joggers.html",
                "https://www2.hm.com/en_us/women/sport/matching-sets.html",
                "https://www2.hm.com/en_us/women/sport/shorts.html",
                "https://www2.hm.com/en_us/women/sport/dresses.html",
                "https://www2.hm.com/en_us/women/sport/yoga.html",
                "https://www2.hm.com/en_us/women/sport/running.html",
                "https://www2.hm.com/en_us/women/sport/gym.html",
                "https://www2.hm.com/en_us/women/sport/gym/tops.html",
                "https://www2.hm.com/en_us/women/sport/gym/hoodies-sweatshirts.html",
                "https://www2.hm.com/en_us/women/sport/gym/jackets.html",
                "https://www2.hm.com/en_us/women/sport/gym/leggings.html",
                "https://www2.hm.com/en_us/women/sport/gym/shorts.html",
                "https://www2.hm.com/en_us/women/sport/gym/accessories.html",
                "https://www2.hm.com/en_us/women/sport/yoga/leggings.html",
                "https://www2.hm.com/en_us/women/sport/yoga/bras.html",
            ]
            
            logger.info(f"Found {len(categories)} categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def get_product_urls(self, category_url):
        try:
            logger.info(f"Fetching category page: {category_url}")
            response = self.session.get(category_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all product links
            product_urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'productpage' in href:
                    full_url = self.base_url + href if href.startswith('/') else href
                    if full_url not in product_urls:
                        product_urls.append(full_url)
                        
            logger.info(f"Found {len(product_urls)} products in category")
            return product_urls

        except Exception as e:
            logger.error(f"Error fetching category page: {e}")
            return []

    def scrape_product(self, url):
        try:
            time.sleep(1)
            logger.info(f"Scraping product: {url}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get JSON-LD data
            script = soup.find('script', {'type': 'application/ld+json'})
            if not script or not script.string:
                logger.warning(f"No JSON-LD data found for {url}")
                return None
                
            product_data = json.loads(script.string)
            
            images = []
            if isinstance(product_data.get('image'), list):
                images = product_data['image']
            elif product_data.get('image'):
                images = [product_data['image']]

            # Get colors
            colors = []
            color = product_data.get('color')
            if color:
                colors = [c.strip() for c in color.split('/')]

            # Get price
            price = None
            currency = None
            if product_data.get('offers'):
                if isinstance(product_data['offers'], list):
                    offer = product_data['offers'][0]
                else:
                    offer = product_data['offers']
                price = offer.get('price')
                currency = offer.get('priceCurrency')

            product_info = {
                'url': url,
                'name': product_data.get('name'),
                'brand': product_data.get('brand', {}).get('name'),
                'description': product_data.get('description'),
                'colors': colors,
                'sku': product_data.get('sku'),
                'price': price,
                'currency': currency,
                'images': images,
                'category': product_data.get('category', {}).get('name')
            }
            
            logger.info(f"Successfully scraped: {product_info['name']}")
            return product_info

        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None

def main():
    scraper = HMScraper()
    
    # Get women's categories
    categories = scraper.get_women_categories()
    logger.info(f"Starting to scrape {len(categories)} categories")
    
    all_products = []
    for category_url in categories:
        logger.info(f"\nProcessing category: {category_url}")
        
        # Get all product URLs for this category
        product_urls = scraper.get_product_urls(category_url)
        
        # Scrape each product
        for url in product_urls:
            product_data = scraper.scrape_product(url)
            if product_data:
                all_products.append(product_data)
                
                # Save progress every 10 products
                if len(all_products) % 10 == 0:
                    with open('hm_women_products_progress.json', 'w', encoding='utf-8') as f:
                        json.dump(all_products, f, indent=2, ensure_ascii=False)
                    logger.info(f"Progress saved: {len(all_products)} products")
    
    # Save final results
    if all_products:
        with open('hm_women_products_final.json', 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)
        logger.info(f"\nFinished! Saved {len(all_products)} products to hm_women_products_final.json")
    else:
        logger.error("No products were successfully scraped")

if __name__ == "__main__":
    main()
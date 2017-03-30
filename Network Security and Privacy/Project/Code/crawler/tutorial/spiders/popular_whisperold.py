import scrapy

class DmozSpider(scrapy.Spider):
    name = "popularold"
    #allowed_domains = ["dmoz.org"]
    start_urls = [
        #"https://whisper.sh/"
        "https://whisper.sh/whispers/popular?wid=052caf88c3a99eb4788dcdc1f77015d8d869be"
        "https://whisper.sh/whispers/popular?wid=052caeb80237af1d76bb956d76e0ed9e60474b"
        "https://whisper.sh/whispers/popular?wid=052caea0e75a6b5dafb46bfe53159035332077"
        "https://whisper.sh/whispers/popular?wid=052c7cd596971cd93b3275c2ea36c8d6476803"
        "https://whisper.sh/whispers/popular?wid=052caf79cddf32b3cddb4ecdb4c30f2f41d237"
        "https://whisper.sh/whispers/popular?wid=052c9f38debb39df02f3c2e4eb12509d9b5e2f"
        "https://whisper.sh/whispers/popular?wid=052cb1c55ddbc5c951fe03ccd1cc08058fabd6"
        "https://whisper.sh/whispers/popular?wid=052c8c336a3dbc7362cb668094b55824498c95"
    ]

    def parse(self, response):
        filename = response.url.split("?")[-1] + '.html'
        with open(filename, 'wb') as f:
            #f.write(response.body)
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]/div[@class="grid-item"]').extract())) #extract and write div
            divs=response.xpath('//div[@class="grid-item"]')#extract div in variable
            #imgsrc=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()
            pop_content=response.xpath('//div[@class="grid-item"]//a[@class="whisper-img ga"]//meta/@content').extract()
            pop_likes=response.xpath('//div[@class="grid-item"]//div[@class="whisper-meta"]//ul/li/text()').extract()
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()))
            #f.write('\n'.join(map(str,divs))) #write extracted div in file divided by newline
            for i in range(len(divs)):
            	#f.write(imgsrc[i]+'###')
            # ///// strip new lines from content and write to file
                contenttemp=str(pop_content[i])
            	contenttemp=contenttemp.replace('\n', ' ').replace('\r', '')

            	f.write(contenttemp+'###')
            	f.write(pop_likes[i]+'###')
            	f.write(pop_likes[i+1]+'\n\n')#+' '+content[i]+' '+likes[i]+' '+like[i+1]+'\n')
            
          


            #for d in divs:
            #	f.write(d+'\n')
           	#f.write('\n')
			
         
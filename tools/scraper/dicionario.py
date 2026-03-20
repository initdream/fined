import os
import time
import requests
from bs4 import BeautifulSoup


PASTA_DESTINO = "artigos_dicionario_financeiro"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

if not os.path.exists(PASTA_DESTINO):
    os.makedirs(PASTA_DESTINO)


def obter_todas_as_urls_do_sitemap():
    #sitemap.xml
    url_sitemap = "https://www.dicionariofinanceiro.com/sitemap.xml?source=articles&page=1" 
    resposta = requests.get(url_sitemap, headers=HEADERS)
    soup = BeautifulSoup(resposta.content, 'xml') # Usamos 'xml' em vez de 'html.parser'
    
    links = []
    for loc in soup.find_all("loc"):
        link = loc.get_text(strip=True)
        links.append(link)
        
    return links


def converter_tabela_para_markdown(tabela_html):
    linhas_markdown = []
    linhas = tabela_html.find_all('tr')
    
    if not linhas:
        return ""

    for i, linha in enumerate(linhas):
        celulas = linha.find_all(['th', 'td'])
        textos_celulas = [
            celula.get_text(strip=True).replace('\n', ' ').replace('|', '\\|') 
            for celula in celulas
        ]
        if not textos_celulas:
            continue
        linha_md = "| " + " | ".join(textos_celulas) + " |"
        linhas_markdown.append(linha_md)
        if i == 0:
            separador = "| " + " | ".join(["---"] * len(textos_celulas)) + " |"
            linhas_markdown.append(separador)
    return "\n" + "\n".join(linhas_markdown) + "\n"

def extrair_artigo(url):
    try:
        resposta = requests.get(url, headers=HEADERS)

        if resposta.status_code != 200:
            print(f"[ERRO] Falha ao acessar {url} (Status: {resposta.status_code})")
            return
        soup = BeautifulSoup(resposta.content, 'html.parser')
        h1 = soup.find('h1')
        titulo = h1.get_text(strip=True) if h1 else "Artigo_Sem_Titulo"
        nome_arquivo = "".join(c for c in titulo if c.isalnum() or c in " -_").strip()
        
        texto_completo = f"TÍTULO: {titulo}\nURL: {url}\n\n"
        container = soup.find('div', id="articleBody")
        
        if container:
            # remover botoes redes sociais
            divs_compartilhar = container.find_all('div', class_='share-module')
            for div in divs_compartilhar:
                div.decompose() 
            todas_uls = container.find_all('ul')
            if todas_uls:
                ultima_ul = todas_uls[-1]
                if ultima_ul.find('a'):
                    for irmao in ultima_ul.find_next_siblings():
                        irmao.decompose()
                    ultima_ul.decompose()
            tabelas = container.find_all('table')
            for tabela in tabelas:
                tabela_md = converter_tabela_para_markdown(tabela)
                tabela.replace_with(tabela_md)
            texto_artigo = container.get_text(separator='\n\n', strip=True)
            texto_completo += texto_artigo
        else:
            print(f"[AVISO] A div 'articleBody' não foi encontrada na URL: {url}")
            texto_completo += "\n[ERRO] O conteúdo principal não foi encontrado nesta página."


        caminho = os.path.join(PASTA_DESTINO, f"{nome_arquivo}.txt")
        with open(caminho, "w", encoding="utf-8") as arquivo:
            arquivo.write(texto_completo)
            
        print(f"[SUCESSO] Baixado: {titulo}")

    except Exception as e:
        print(f"[ERRO NO CÓDIGO] Falha em {url}: {e}")

def executar_scraper():
    urls_para_baixar = obter_todas_as_urls_do_sitemap()
    extensoes_ignoradas = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.pdf', '.mp4')

    print(f"Iniciando o download de {len(urls_para_baixar)} artigos...\n")

    for url in urls_para_baixar:
        if url.lower().endswith(extensoes_ignoradas):
            print(f"[IGNORADO] Arquivo de mídia pulado: {url}")
            continue 
        extrair_artigo(url)
        #time.sleep(3) 

    print("\nFinalizado! Verifique a pasta:", PASTA_DESTINO)

if __name__ == "__main__":
    executar_scraper()

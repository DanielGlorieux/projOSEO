"""
Script d'indexation pour la base de connaissances Pinecone
Exécuter ce script pour indexer la documentation ONEA dans Pinecone
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader, PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Charger les variables d'environnement
load_dotenv()

# Configuration
DOCS_PATH = Path(__file__).parent / "docs"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "onea-knowledge-base")

def load_documents():
    """Charger tous les documents du dossier docs/"""
    print("📂 Chargement des documents...")
    
    documents = []
    
    # Charger les fichiers texte et markdown
    txt_loader = DirectoryLoader(
        str(DOCS_PATH),
        glob="**/*.{txt,md}",
        loader_cls=TextLoader,
        show_progress=True
    )
    documents.extend(txt_loader.load())
    
    # Charger les PDFs si possible
    try:
        pdf_loader = DirectoryLoader(
            str(DOCS_PATH),
            glob="**/*.pdf",
            loader_cls=PDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"⚠️ Impossible de charger les PDFs: {e}")
    
    print(f"✅ {len(documents)} documents chargés")
    return documents


def split_documents(documents):
    """Découper les documents en chunks"""
    print("✂️ Découpage des documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"✅ {len(texts)} chunks créés")
    return texts


def create_embeddings():
    """Créer l'instance d'embeddings"""
    print("🔮 Initialisation des embeddings...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY manquante dans .env")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    
    print("✅ Embeddings initialisés")
    return embeddings


def index_documents(texts, embeddings):
    """Indexer les documents dans Pinecone"""
    print(f"🗂️ Indexation dans Pinecone (index: {INDEX_NAME})...")
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY manquante dans .env")
    
    # Vérifier que l'index existe
    pc = Pinecone(api_key=api_key)
    indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in indexes:
        print(f"❌ Index '{INDEX_NAME}' n'existe pas dans Pinecone!")
        print(f"📝 Indexes disponibles: {indexes}")
        print("\n💡 Créer l'index sur https://app.pinecone.io/ avec:")
        print("   - Nom: onea-knowledge-base")
        print("   - Dimension: 768")
        print("   - Métrique: cosine")
        return False
    
    # Indexer les documents
    vectorstore = PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=INDEX_NAME
    )
    
    print(f"✅ {len(texts)} documents indexés avec succès!")
    return True


def main():
    """Fonction principale"""
    print("=" * 60)
    print("🚀 Indexation Base de Connaissances ONEA")
    print("=" * 60)
    
    try:
        # Vérifier que le dossier docs existe
        if not DOCS_PATH.exists():
            print(f"❌ Dossier {DOCS_PATH} non trouvé!")
            print("💡 Créer le dossier et y placer la documentation ONEA")
            DOCS_PATH.mkdir(parents=True, exist_ok=True)
            print(f"✅ Dossier {DOCS_PATH} créé")
            print("\n📝 Placer vos documents (.txt, .md, .pdf) dans ce dossier")
            print("   puis relancer ce script")
            return
        
        # Vérifier qu'il y a des documents
        docs_count = len(list(DOCS_PATH.glob("**/*.{txt,md,pdf}")))
        if docs_count == 0:
            print(f"❌ Aucun document trouvé dans {DOCS_PATH}")
            print("💡 Ajouter des fichiers .txt, .md ou .pdf")
            return
        
        # Pipeline d'indexation
        documents = load_documents()
        if not documents:
            print("❌ Aucun document n'a pu être chargé")
            return
        
        texts = split_documents(documents)
        embeddings = create_embeddings()
        success = index_documents(texts, embeddings)
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 Indexation terminée avec succès!")
            print("=" * 60)
            print("\n💡 Prochaines étapes:")
            print("   1. Démarrer l'API: python api/main.py")
            print("   2. Démarrer le frontend: cd dashboard/react-app && npm run dev")
            print("   3. Tester le chatbot sur http://localhost:3000")
        else:
            print("\n❌ L'indexation a échoué")
    
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        print("\n💡 Vérifier:")
        print("   - Fichier .env existe avec GOOGLE_API_KEY et PINECONE_API_KEY")
        print("   - Index Pinecone créé (dimension: 768, metric: cosine)")
        print("   - Connexion internet active")
        raise


if __name__ == "__main__":
    main()

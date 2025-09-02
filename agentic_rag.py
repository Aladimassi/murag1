import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass, field
import re
from pathlib import Path
import time
from abc import ABC, abstractmethod
from enum import Enum
import logging
from datetime import datetime
import random
from collections import defaultdict, deque
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types de requêtes pour classification"""
    SIMPLE_QA = "simple_qa"
    COMPLEX_ANALYSIS = "complex_analysis"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    MULTI_DOC = "multi_document"
    FACTUAL = "factual"
    REASONING = "reasoning"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    EXPLORATION = "exploration"
    IMAGE_ANALYSIS = "image_analysis"
    VISUAL_QA = "visual_qa"
    OCR_EXTRACTION = "ocr_extraction"
    MULTIMODAL = "multimodal"

class ConfidenceLevel(Enum):
    """Niveaux de confiance pour les décisions agentiques"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class AgentAction:
    """Représente une action effectuée par l'agent"""
    action_type: str
    timestamp: datetime
    confidence: float
    reasoning: str
    success: bool = None
    feedback_score: float = None

@dataclass
class Document:
    """Classe pour représenter un document avec métadonnées (support multi-modal)"""
    content: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None
    doc_type: str = "text"  # "text", "image", "mixed"
    image_data: Optional[bytes] = None
    image_description: Optional[str] = None

@dataclass
class ConversationMemory:
    """Mémoire de conversation pour maintenir le contexte"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    action_history: List[AgentAction] = field(default_factory=list)
    performance_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
    
    def add_action(self, action: AgentAction):
        """Ajoute une action à l'historique pour l'apprentissage"""
        self.action_history.append(action)
        
        # Mise à jour des métriques de performance
        if action.success is not None:
            self.performance_metrics[action.action_type].append(1.0 if action.success else 0.0)
    
    def get_recent_context(self, max_messages: int = 5) -> str:
        """Récupère le contexte récent de la conversation"""
        recent = self.messages[-max_messages:]
        context_parts = []
        for msg in recent:
            context_parts.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(context_parts)
    
    def get_performance_for_action(self, action_type: str) -> float:
        """Retourne le taux de succès pour un type d'action"""
        if action_type not in self.performance_metrics:
            return 0.5  # Défaut neutre
        
        successes = self.performance_metrics[action_type]
        if not successes:
            return 0.5
        
        return sum(successes) / len(successes)
    
    def should_try_action(self, action_type: str, threshold: float = 0.3) -> bool:
        """Détermine si un type d'action devrait être tenté basé sur les performances passées"""
        performance = self.get_performance_for_action(action_type)
        return performance > threshold

class Tool(ABC):
    """Classe de base pour tous les outils"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass

class DocumentSearchTool(Tool):
    """Outil de recherche dans les documents"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, query: str, k: int = 5) -> Dict[str, Any]:
        docs = self.rag_system.search_similar(query, k)
        return {
            "tool": "document_search",
            "results": docs,
            "count": len(docs)
        }
    
    def get_description(self) -> str:
        return "Recherche de documents similaires dans la base de connaissances"

class SummarizationTool(Tool):
    """Outil de résumé de documents"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, documents: List[Document], focus: str = None) -> Dict[str, Any]:
        # Combine le contenu des documents
        combined_content = "\n\n".join([doc.content for doc in documents])
        
        prompt = f"""Résume le contenu suivant de manière concise et structurée:

{combined_content}

Instructions:
- Fais un résumé clair et structuré
- Identifie les points clés
- Utilise des puces ou une structure organisée
{f"- Focus particulier sur: {focus}" if focus else ""}
"""
        
        response = self.rag_system.query_gemini(prompt)
        return {
            "tool": "summarization",
            "summary": response,
            "source_count": len(documents)
        }
    
    def get_description(self) -> str:
        return "Résume le contenu de plusieurs documents"

class ComparisonTool(Tool):
    """Outil de comparaison entre documents"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        # Prépare le contenu pour comparaison
        doc_contents = []
        for i, doc in enumerate(documents):
            doc_contents.append(f"Document {i+1}: {doc.content}")
        
        combined = "\n\n".join(doc_contents)
        
        prompt = f"""Compare les informations dans ces documents concernant: {query}

{combined}

Instructions:
- Identifie les similitudes et différences
- Structure ta réponse clairement
- Mentionne les sources spécifiques
- Tire des conclusions si pertinent
"""
        
        response = self.rag_system.query_gemini(prompt)
        return {
            "tool": "comparison",
            "comparison": response,
            "documents_compared": len(documents)
        }
    
    def get_description(self) -> str:
        return "Compare les informations entre plusieurs documents"

class AnalysisTool(Tool):
    """Outil d'analyse approfondie"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, query: str, documents: List[Document], analysis_type: str = "general") -> Dict[str, Any]:
        combined_content = "\n\n".join([doc.content for doc in documents])
        
        analysis_prompts = {
            "general": "Fais une analyse détaillée du contenu",
            "critical": "Fais une analyse critique en identifiant les forces et faiblesses",
            "trend": "Identifie les tendances et patterns dans le contenu",
            "implications": "Analyse les implications et conséquences"
        }
        
        analysis_instruction = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        prompt = f"""{analysis_instruction} concernant: {query}

Contenu à analyser:
{combined_content}

Instructions:
- Fournis une analyse structurée et approfondie
- Utilise des arguments basés sur les données
- Identifie les insights clés
- Propose des recommandations si approprié
"""
        
        response = self.rag_system.query_gemini(prompt)
        return {
            "tool": "analysis",
            "analysis": response,
            "type": analysis_type,
            "source_count": len(documents)
        }
    
    def get_description(self) -> str:
        return "Effectue une analyse approfondie du contenu"

class SelfReflectionTool(Tool):
    """Outil d'auto-réflexion pour évaluer et améliorer les réponses"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        reflection_prompt = f"""Évalue la qualité de cette réponse et propose des améliorations:

Question originale: {query}

Réponse fournie: {answer}

Contexte utilisé: {context[:1000]}...

Critères d'évaluation:
1. Pertinence par rapport à la question
2. Utilisation appropriée du contexte
3. Clarté et structure de la réponse
4. Complétude de l'information
5. Suggestions d'amélioration

Fournis:
- Score de qualité (1-10)
- Points forts
- Points faibles
- Recommandations d'amélioration
"""
        
        reflection = self.rag_system.query_gemini(reflection_prompt)
        
        # Extraction simple du score (amélioration possible avec regex)
        score = 7.0  # Défaut
        if "score" in reflection.lower():
            try:
                import re
                score_match = re.search(r'score.*?(\d+(?:\.\d+)?)', reflection.lower())
                if score_match:
                    score = float(score_match.group(1))
            except:
                pass
        
        return {
            "tool": "self_reflection",
            "reflection": reflection,
            "quality_score": score,
            "needs_improvement": score < 6.0
        }
    
    def get_description(self) -> str:
        return "Évalue et améliore la qualité des réponses par auto-réflexion"

class ContextEnrichmentTool(Tool):
    """Outil d'enrichissement de contexte basé sur l'historique"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, query: str, memory: ConversationMemory) -> Dict[str, Any]:
        """Enrichit le contexte en utilisant l'historique de conversation"""
        recent_context = memory.get_recent_context(3)
        
        if not recent_context:
            return {"tool": "context_enrichment", "enriched_query": query, "enrichment_applied": False}
        
        enrichment_prompt = f"""Analyse cette conversation et enrichis la requête actuelle avec le contexte pertinent:

Historique récent:
{recent_context}

Requête actuelle: {query}

Instructions:
- Identifie les éléments de contexte pertinents
- Reformule la requête pour inclure le contexte implicite
- Conserve l'intention originale
- Ne dépasse pas 200 mots

Requête enrichie:"""
        
        enriched_query = self.rag_system.query_gemini(enrichment_prompt)
        
        return {
            "tool": "context_enrichment",
            "original_query": query,
            "enriched_query": enriched_query,
            "enrichment_applied": len(enriched_query) > len(query)
        }
    
    def get_description(self) -> str:
        return "Enrichit les requêtes avec le contexte conversationnel"

class ImageAnalysisTool(Tool):
    """Outil d'analyse d'images avec Gemini Vision"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, image_path: str = None, image_data: bytes = None, query: str = "Décris cette image en détail") -> Dict[str, Any]:
        """Analyse une image avec Gemini Vision"""
        
        try:
            # Préparation de l'image
            if image_path:
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
            
            if not image_data:
                return {"tool": "image_analysis", "error": "Aucune image fournie"}
            
            # Conversion en base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Détection du format d'image
            image_format = self._detect_image_format(image_data)
            
            # Appel à Gemini Vision
            description = self._analyze_with_gemini_vision(image_base64, image_format, query)
            
            return {
                "tool": "image_analysis",
                "description": description,
                "image_format": image_format,
                "query_used": query,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse image: {e}")
            return {
                "tool": "image_analysis", 
                "error": str(e),
                "success": False
            }
    
    def _detect_image_format(self, image_data: bytes) -> str:
        """Détecte le format de l'image"""
        try:
            image = Image.open(io.BytesIO(image_data))
            return image.format.lower() if image.format else "unknown"
        except:
            return "unknown"
    
    def _analyze_with_gemini_vision(self, image_base64: str, image_format: str, query: str) -> str:
        """Analyse l'image avec Gemini Vision avec retry et meilleure gestion d'erreurs"""
        
        # URL pour Gemini Vision
        vision_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        headers = {"Content-Type": "application/json"}
        
        # Détermination du type MIME
        mime_type = f"image/{image_format}" if image_format != "unknown" else "image/jpeg"
        
        data = {
            "contents": [{
                "parts": [
                    {"text": query},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 1024,
            }
        }
        
        # Retry logic avec backoff exponentiel
        max_retries = 3
        base_timeout = 15
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt)  # 15s, 30s, 60s
                print(f"🔄 Tentative {attempt + 1}/{max_retries} d'analyse image (timeout: {timeout}s)")
                
                response = requests.post(
                    f"{vision_url}?key={self.rag_system.api_key}",
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
                
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    print("✅ Analyse d'image réussie")
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    print("⚠️ Aucune description générée par l'API")
                    return "Erreur: Aucune description générée pour l'image"
                    
            except requests.exceptions.Timeout as e:
                print(f"⏱️ Timeout à la tentative {attempt + 1}: {timeout}s dépassé")
                if attempt == max_retries - 1:
                    return f"❌ Timeout après {max_retries} tentatives. Vérifiez votre connexion internet."
                time.sleep(2 ** attempt)  # Attente avant retry
                
            except requests.exceptions.ConnectionError as e:
                print(f"🔌 Erreur de connexion à la tentative {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return f"❌ Impossible de se connecter à l'API Gemini après {max_retries} tentatives. Vérifiez votre connexion internet et votre clé API."
                time.sleep(2 ** attempt)
                
            except requests.exceptions.RequestException as e:
                print(f"🚫 Erreur de requête à la tentative {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return f"❌ Erreur API Gemini après {max_retries} tentatives: {str(e)}"
                time.sleep(1)
                
            except (KeyError, IndexError) as e:
                print(f"📋 Erreur de format de réponse: {e}")
                return f"❌ Format de réponse API inattendu: {str(e)}"
        
        return "❌ Échec de toutes les tentatives d'analyse"
    
    def get_description(self) -> str:
        return "Analyse et décrit le contenu des images avec Gemini Vision"

class OCRTool(Tool):
    """Outil d'extraction de texte OCR des images"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, image_path: str = None, image_data: bytes = None) -> Dict[str, Any]:
        """Extrait le texte d'une image avec OCR"""
        
        try:
            # Utilisation de Gemini Vision pour l'OCR
            ocr_query = """Extrait tout le texte visible dans cette image. 
            Conserve la structure et la mise en forme autant que possible.
            Si il n'y a pas de texte, indique 'Aucun texte détecté'."""
            
            # Réutilisation de l'outil d'analyse d'image pour l'OCR
            image_tool = ImageAnalysisTool(self.rag_system)
            result = image_tool.execute(image_path=image_path, image_data=image_data, query=ocr_query)
            
            if result.get("success", False):
                extracted_text = result["description"]
                
                # Nettoyage du texte extrait
                cleaned_text = self._clean_extracted_text(extracted_text)
                
                return {
                    "tool": "ocr",
                    "extracted_text": cleaned_text,
                    "raw_text": extracted_text,
                    "has_text": "aucun texte" not in cleaned_text.lower(),
                    "success": True
                }
            else:
                return {
                    "tool": "ocr",
                    "error": result.get("error", "Erreur OCR"),
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return {
                "tool": "ocr",
                "error": str(e),
                "success": False
            }
    
    def _clean_extracted_text(self, text: str) -> str:
        """Nettoie le texte extrait par OCR"""
        if not text:
            return ""
        
        # Nettoyage basique
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)  # Saut de ligne après ponctuation
        
        return text
    
    def get_description(self) -> str:
        return "Extrait le texte des images par reconnaissance optique (OCR)"

class MultimodalAnalysisTool(Tool):
    """Outil d'analyse multimodale combinant texte et images"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def execute(self, query: str, text_documents: List[Document] = None, 
               image_documents: List[Document] = None) -> Dict[str, Any]:
        """Analyse combinée de texte et d'images"""
        
        try:
            analysis_parts = []
            
            # Analyse du texte
            if text_documents:
                text_content = "\n\n".join([doc.content for doc in text_documents])
                text_analysis = f"Contenu textuel pertinent:\n{text_content[:2000]}..."
                analysis_parts.append(text_analysis)
            
            # Analyse des images
            if image_documents:
                image_tool = ImageAnalysisTool(self.rag_system)
                
                for i, img_doc in enumerate(image_documents[:3]):  # Limite à 3 images
                    if img_doc.image_data:
                        img_result = image_tool.execute(
                            image_data=img_doc.image_data,
                            query=f"Analyse cette image dans le contexte de: {query}"
                        )
                        if img_result.get("success", False):
                            analysis_parts.append(f"Image {i+1}: {img_result['description']}")
            
            # Synthèse multimodale
            if analysis_parts:
                combined_analysis = "\n\n".join(analysis_parts)
                
                synthesis_prompt = f"""Effectue une analyse multimodale en combinant les informations textuelles et visuelles:

Question: {query}

Informations disponibles:
{combined_analysis}

Instructions:
- Synthétise les informations textuelles et visuelles
- Identifie les correspondances et complémentarités
- Fournis une réponse intégrée et cohérente
- Mentionne spécifiquement les éléments visuels pertinents
"""
                
                synthesis = self.rag_system.query_gemini(synthesis_prompt)
                
                return {
                    "tool": "multimodal_analysis",
                    "synthesis": synthesis,
                    "text_sources": len(text_documents) if text_documents else 0,
                    "image_sources": len(image_documents) if image_documents else 0,
                    "success": True
                }
            else:
                return {
                    "tool": "multimodal_analysis",
                    "error": "Aucune source multimodale disponible",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Erreur analyse multimodale: {e}")
            return {
                "tool": "multimodal_analysis",
                "error": str(e),
                "success": False
            }
    
    def get_description(self) -> str:
        return "Analyse combinée de contenu textuel et visuel pour une compréhension multimodale"

class QueryClassifier:
    """Classifie les requêtes pour déterminer la stratégie optimale avec apprentissage adaptatif"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.classification_history = deque(maxlen=100)  # Historique des classifications
        self.pattern_weights = defaultdict(float)  # Poids appris pour les patterns
    
    def classify_query(self, query: str, memory: ConversationMemory = None) -> Dict[str, Any]:
        """Classifie une requête et détermine la stratégie avec apprentissage adaptatif"""
        
        # Classification basée sur les mots-clés (améliorée)
        basic_classification = self._basic_keyword_classification(query)
        
        # Classification contextuelle si mémoire disponible
        contextual_boost = self._contextual_classification(query, memory) if memory else {}
        
        # Classification sémantique basée sur l'IA
        semantic_classification = self._semantic_classification(query)
        
        # Fusion des classifications avec pondération adaptative
        final_classification = self._fuse_classifications(
            basic_classification, contextual_boost, semantic_classification
        )
        
        # Apprentissage des patterns
        self._learn_from_classification(query, final_classification)
        
        return final_classification
    
    def _basic_keyword_classification(self, query: str) -> Dict[str, Any]:
        """Classification basique par mots-clés avec pondération adaptative"""
        
        # Mots-clés étendus avec poids dynamiques
        keyword_patterns = {
            QueryType.SUMMARY: {
                "keywords": ["résume", "résumé", "synthèse", "principal", "essentiel", "grandes lignes", "aperçu"],
                "weight": self.pattern_weights.get("summary", 1.0)
            },
            QueryType.COMPARISON: {
                "keywords": ["compare", "différence", "similitude", "versus", "entre", "contraste", "opposition"],
                "weight": self.pattern_weights.get("comparison", 1.0)
            },
            QueryType.COMPLEX_ANALYSIS: {
                "keywords": ["analyse", "pourquoi", "comment", "impact", "conséquence", "implications", "causes"],
                "weight": self.pattern_weights.get("analysis", 1.0)
            },
            QueryType.FACTUAL: {
                "keywords": ["qui", "quoi", "quand", "où", "combien", "définition", "qu'est-ce que"],
                "weight": self.pattern_weights.get("factual", 1.0)
            },
            QueryType.FOLLOW_UP: {
                "keywords": ["également", "aussi", "de plus", "continuer", "approfondir", "préciser"],
                "weight": self.pattern_weights.get("follow_up", 1.0)
            },
            QueryType.IMAGE_ANALYSIS: {
                "keywords": ["image", "photo", "illustration", "figure", "graphique", "schéma", "diagramme", "voir", "montre"],
                "weight": self.pattern_weights.get("image_analysis", 1.0) 
            },
            QueryType.VISUAL_QA: {
                "keywords": ["dans l'image", "sur la photo", "visible", "aperçoit", "regarde", "observe"],
                "weight": self.pattern_weights.get("visual_qa", 1.0)
            },
            QueryType.OCR_EXTRACTION: {
                "keywords": ["texte", "écrit", "inscription", "lecture", "lire", "transcription"],
                "weight": self.pattern_weights.get("ocr_extraction", 1.0)
            },
            QueryType.MULTIMODAL: {
                "keywords": ["combine", "ensemble", "à la fois", "texte et image", "document complet"],
                "weight": self.pattern_weights.get("multimodal", 1.0)
            }
        }
        
        query_lower = query.lower()
        scores = {}
        
        for query_type, pattern in keyword_patterns.items():
            score = 0.0
            for keyword in pattern["keywords"]:
                if keyword in query_lower:
                    score += pattern["weight"]
            
            if score > 0:
                scores[query_type] = score * pattern["weight"]
        
        if not scores:
            return {
                "type": QueryType.SIMPLE_QA,
                "confidence": 0.5,
                "strategy": "standard_rag",
                "reasoning": "Aucun pattern spécifique détecté"
            }
        
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 3.0, 0.9)  # Normalisation
        
        strategy_map = {
            QueryType.SUMMARY: "summarization_focused",
            QueryType.COMPARISON: "comparison_focused", 
            QueryType.COMPLEX_ANALYSIS: "analysis_focused",
            QueryType.FACTUAL: "search_focused",
            QueryType.FOLLOW_UP: "context_enhanced",
            QueryType.IMAGE_ANALYSIS: "image_analysis_focused",
            QueryType.VISUAL_QA: "visual_qa_focused",
            QueryType.OCR_EXTRACTION: "ocr_focused",
            QueryType.MULTIMODAL: "multimodal_focused"
        }
        
        return {
            "type": best_type,
            "confidence": confidence,
            "strategy": strategy_map.get(best_type, "standard_rag"),
            "reasoning": f"Détection de mots-clés pour {best_type.value}"
        }
    
    def _contextual_classification(self, query: str, memory: ConversationMemory) -> Dict[str, Any]:
        """Classification basée sur le contexte de conversation"""
        if not memory.messages:
            return {}
        
        recent_messages = memory.messages[-3:]
        context_indicators = []
        
        for msg in recent_messages:
            if msg["role"] == "user":
                content = msg["content"].lower()
                if any(word in content for word in ["continue", "aussi", "également", "de plus"]):
                    context_indicators.append("follow_up")
                if any(word in content for word in ["précise", "détaille", "explique"]):
                    context_indicators.append("clarification")
        
        if context_indicators:
            most_common = max(set(context_indicators), key=context_indicators.count)
            return {
                "contextual_boost": most_common,
                "confidence_boost": 0.2
            }
        
        return {}
    
    def _semantic_classification(self, query: str) -> Dict[str, Any]:
        """Classification sémantique utilisant l'IA"""
        semantic_prompt = f"""Analyse cette question et classifie-la selon ces catégories:

Question: "{query}"

Catégories possibles:
1. SIMPLE_QA - Question simple nécessitant une réponse directe
2. SUMMARY - Demande de résumé ou synthèse
3. COMPARISON - Comparaison entre éléments
4. ANALYSIS - Analyse approfondie ou explicative
5. FACTUAL - Question factuelle précise
6. REASONING - Raisonnement ou argumentation
7. EXPLORATION - Exploration de sujet

Réponds uniquement avec:
- Catégorie: [CATEGORIE]
- Confiance: [0.1-1.0]
- Raison: [explication courte]"""
        
        try:
            response = self.rag_system.query_gemini(semantic_prompt)
            
            # Parsing simple de la réponse (amélioration possible)
            if "SUMMARY" in response.upper():
                return {"semantic_type": QueryType.SUMMARY, "semantic_confidence": 0.7}
            elif "COMPARISON" in response.upper():
                return {"semantic_type": QueryType.COMPARISON, "semantic_confidence": 0.7}
            elif "ANALYSIS" in response.upper():
                return {"semantic_type": QueryType.COMPLEX_ANALYSIS, "semantic_confidence": 0.7}
            elif "FACTUAL" in response.upper():
                return {"semantic_type": QueryType.FACTUAL, "semantic_confidence": 0.7}
            
        except Exception as e:
            logger.warning(f"Erreur classification sémantique: {e}")
        
        return {}
    
    def _fuse_classifications(self, basic: Dict, contextual: Dict, semantic: Dict) -> Dict[str, Any]:
        """Fusionne les différentes classifications avec pondération intelligente"""
        
        # Classification de base
        final_type = basic["type"]
        final_confidence = basic["confidence"]
        final_strategy = basic["strategy"]
        reasoning_parts = [basic["reasoning"]]
        
        # Boost contextuel
        if "confidence_boost" in contextual:
            final_confidence = min(final_confidence + contextual["confidence_boost"], 0.95)
            reasoning_parts.append("Contexte conversationnel pris en compte")
        
        # Classification sémantique
        if "semantic_type" in semantic and semantic["semantic_confidence"] > 0.6:
            # Si la classification sémantique est confiante et différente
            if semantic["semantic_type"] != final_type and semantic["semantic_confidence"] > final_confidence:
                final_type = semantic["semantic_type"]
                final_confidence = (final_confidence + semantic["semantic_confidence"]) / 2
                reasoning_parts.append("Classification sémantique privilégiée")
        
        # Mise à jour de la stratégie
        strategy_map = {
            QueryType.SUMMARY: "summarization_focused",
            QueryType.COMPARISON: "comparison_focused",
            QueryType.COMPLEX_ANALYSIS: "analysis_focused",
            QueryType.FACTUAL: "search_focused",
            QueryType.FOLLOW_UP: "context_enhanced",
            QueryType.EXPLORATION: "multi_tool_exploration"
        }
        
        final_strategy = strategy_map.get(final_type, "standard_rag")
        
        return {
            "type": final_type,
            "confidence": final_confidence,
            "strategy": final_strategy,
            "reasoning": " | ".join(reasoning_parts),
            "classification_details": {
                "basic": basic,
                "contextual": contextual,
                "semantic": semantic
            }
        }
    
    def _learn_from_classification(self, query: str, classification: Dict[str, Any]):
        """Apprend des classifications pour améliorer les futures prédictions"""
        
        self.classification_history.append({
            "query": query,
            "classification": classification,
            "timestamp": datetime.now()
        })
        
        # Apprentissage simple des poids (amélioration possible avec ML)
        query_type = classification["type"]
        confidence = classification["confidence"]
        
        if confidence > 0.8:  # Classifications confiantes
            pattern_key = query_type.value
            self.pattern_weights[pattern_key] = min(self.pattern_weights[pattern_key] + 0.1, 2.0)
        elif confidence < 0.4:  # Classifications peu confiantes
            pattern_key = query_type.value
            self.pattern_weights[pattern_key] = max(self.pattern_weights[pattern_key] - 0.05, 0.5)
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de classification pour le monitoring"""
        if not self.classification_history:
            return {"message": "Aucune classification effectuée"}
        
        type_counts = defaultdict(int)
        confidence_sum = 0.0
        
        for record in self.classification_history:
            type_counts[record["classification"]["type"].value] += 1
            confidence_sum += record["classification"]["confidence"]
        
        return {
            "total_classifications": len(self.classification_history),
            "average_confidence": confidence_sum / len(self.classification_history),
            "type_distribution": dict(type_counts),
            "pattern_weights": dict(self.pattern_weights)
        }

class AgentPlanner:
    """Planificateur intelligent pour l'agent avec adaptation et apprentissage"""
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.plan_performance = defaultdict(list)  # Historique de performance des plans
        self.plan_templates = self._initialize_plan_templates()
    
    def _initialize_plan_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialise les templates de plans pour différentes stratégies"""
        return {
            "summarization_focused": [
                {"action": "search", "params": {"query": "{query}", "k": 10}, "priority": 1},
                {"action": "summarize", "params": {"focus": "{query}"}, "priority": 2}
            ],
            "comparison_focused": [
                {"action": "search", "params": {"query": "{query}", "k": 8}, "priority": 1},
                {"action": "compare", "params": {"query": "{query}"}, "priority": 2}
            ],
            "analysis_focused": [
                {"action": "search", "params": {"query": "{query}", "k": 6}, "priority": 1},
                {"action": "analyze", "params": {"query": "{query}", "analysis_type": "general"}, "priority": 2}
            ],
            "context_enhanced": [
                {"action": "enrich_context", "params": {"query": "{query}"}, "priority": 1},
                {"action": "search", "params": {"query": "{enriched_query}", "k": 6}, "priority": 2},
                {"action": "generate", "params": {"query": "{query}"}, "priority": 3}
            ],
            "multi_tool_exploration": [
                {"action": "search", "params": {"query": "{query}", "k": 8}, "priority": 1},
                {"action": "summarize", "params": {"focus": "{query}"}, "priority": 2},
                {"action": "analyze", "params": {"query": "{query}", "analysis_type": "general"}, "priority": 3}
            ],
            "adaptive_deep_dive": [
                {"action": "search", "params": {"query": "{query}", "k": 10}, "priority": 1},
                {"action": "analyze", "params": {"query": "{query}", "analysis_type": "implications"}, "priority": 2},
                {"action": "reflect", "params": {"query": "{query}"}, "priority": 3}
            ],
            "standard_rag": [
                {"action": "search", "params": {"query": "{query}", "k": 5}, "priority": 1},
                {"action": "generate", "params": {"query": "{query}"}, "priority": 2}
            ],
            "image_analysis_focused": [
                {"action": "search_images", "params": {"query": "{query}", "k": 3}, "priority": 1},
                {"action": "analyze_image", "params": {"query": "{query}"}, "priority": 2}
            ],
            "visual_qa_focused": [
                {"action": "search_images", "params": {"query": "{query}", "k": 2}, "priority": 1},
                {"action": "analyze_image", "params": {"query": "{query}"}, "priority": 2},
                {"action": "search", "params": {"query": "{query}", "k": 3}, "priority": 3},
                {"action": "multimodal_analysis", "params": {"query": "{query}"}, "priority": 4}
            ],
            "ocr_focused": [
                {"action": "search_images", "params": {"query": "{query}", "k": 5}, "priority": 1},
                {"action": "extract_text", "params": {"query": "{query}"}, "priority": 2}
            ],
            "multimodal_focused": [
                {"action": "search", "params": {"query": "{query}", "k": 5}, "priority": 1},
                {"action": "search_images", "params": {"query": "{query}", "k": 3}, "priority": 1.5},
                {"action": "multimodal_analysis", "params": {"query": "{query}"}, "priority": 2}
            ]
        }
    
    def create_plan(self, query: str, query_classification: Dict[str, Any], 
                   memory: ConversationMemory) -> List[Dict[str, Any]]:
        """Crée un plan d'exécution adaptatif basé sur la requête et l'historique"""
        
        strategy = query_classification["strategy"]
        confidence = query_classification["confidence"]
        
        # Sélection du template de base
        base_plan = self._get_base_plan(strategy, query)
        
        # Adaptation basée sur la confiance
        adapted_plan = self._adapt_plan_by_confidence(base_plan, confidence)
        
        # Adaptation basée sur l'historique de performance
        optimized_plan = self._optimize_plan_by_history(adapted_plan, strategy, memory)
        
        # Adaptation contextuelle
        final_plan = self._adapt_plan_by_context(optimized_plan, memory)
        
        # Enregistrement du plan pour l'apprentissage
        self._record_plan(final_plan, strategy, query_classification)
        
        return final_plan
    
    def _get_base_plan(self, strategy: str, query: str) -> List[Dict[str, Any]]:
        """Récupère le plan de base pour une stratégie donnée"""
        
        template = self.plan_templates.get(strategy, self.plan_templates["standard_rag"])
        
        # Substitution des variables dans le template
        plan = []
        for step in template:
            step_copy = step.copy()
            if "params" in step_copy:
                params = step_copy["params"].copy()
                for key, value in params.items():
                    if isinstance(value, str) and "{query}" in value:
                        params[key] = value.format(query=query)
                step_copy["params"] = params
            plan.append(step_copy)
        
        return plan
    
    def _adapt_plan_by_confidence(self, plan: List[Dict[str, Any]], confidence: float) -> List[Dict[str, Any]]:
        """Adapte le plan basé sur la confiance de classification"""
        
        adapted_plan = plan.copy()
        
        if confidence < 0.5:
            # Faible confiance : ajouter des étapes de vérification
            verification_step = {
                "action": "search", 
                "params": {"query": "verification", "k": 3}, 
                "priority": 0.5,
                "reasoning": "Vérification due à faible confiance"
            }
            adapted_plan.insert(0, verification_step)
        
        elif confidence > 0.8:
            # Haute confiance : optimiser pour la rapidité
            for step in adapted_plan:
                if step["action"] == "search" and "k" in step["params"]:
                    step["params"]["k"] = min(step["params"]["k"], 5)  # Réduire le nombre de documents
        
        return adapted_plan
    
    def _optimize_plan_by_history(self, plan: List[Dict[str, Any]], strategy: str, 
                                memory: ConversationMemory) -> List[Dict[str, Any]]:
        """Optimise le plan basé sur l'historique de performance"""
        
        # Vérification des performances passées pour cette stratégie
        if strategy in self.plan_performance:
            avg_performance = sum(self.plan_performance[strategy]) / len(self.plan_performance[strategy])
            
            if avg_performance < 0.6:  # Performance faible
                # Ajouter une étape d'analyse supplémentaire
                analysis_step = {
                    "action": "analyze",
                    "params": {"query": "{enhanced_query}", "analysis_type": "critical"},
                    "priority": 1.5,
                    "reasoning": "Analyse supplémentaire due à performance historique faible"
                }
                plan.append(analysis_step)
        
        # Adaptation basée sur les actions qui ont bien fonctionné
        for action_type in ["summarize", "compare", "analyze"]:
            if memory.should_try_action(action_type, threshold=0.7):
                # Cette action a bien fonctionné par le passé, augmenter sa priorité
                for step in plan:
                    if step["action"] == action_type:
                        step["priority"] = step.get("priority", 1) + 0.2
        
        return plan
    
    def _adapt_plan_by_context(self, plan: List[Dict[str, Any]], memory: ConversationMemory) -> List[Dict[str, Any]]:
        """Adapte le plan basé sur le contexte de conversation"""
        
        if len(memory.messages) > 2:
            # Conversation en cours : ajouter enrichissement contextuel si pas déjà présent
            has_context_enrichment = any(step["action"] == "enrich_context" for step in plan)
            
            if not has_context_enrichment:
                context_step = {
                    "action": "enrich_context",
                    "params": {"query": "{original_query}"},
                    "priority": 0.5,
                    "reasoning": "Enrichissement contextuel pour conversation continue"
                }
                plan.insert(0, context_step)
        
        # Tri du plan par priorité
        plan.sort(key=lambda x: x.get("priority", 1))
        
        return plan
    
    def _record_plan(self, plan: List[Dict[str, Any]], strategy: str, classification: Dict[str, Any]):
        """Enregistre le plan pour l'apprentissage futur"""
        
        plan_record = {
            "plan": plan,
            "strategy": strategy,
            "classification": classification,
            "timestamp": datetime.now(),
            "execution_success": None  # Sera mis à jour après l'exécution
        }
        
        # Stockage pour l'apprentissage (simplifié)
        if not hasattr(self, 'plan_history'):
            self.plan_history = deque(maxlen=50)
        self.plan_history.append(plan_record)
    
    def record_plan_performance(self, strategy: str, success_score: float):
        """Enregistre la performance d'un plan pour l'apprentissage"""
        self.plan_performance[strategy].append(success_score)
        
        # Limitation de l'historique
        if len(self.plan_performance[strategy]) > 20:
            self.plan_performance[strategy] = self.plan_performance[strategy][-20:]
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de planification"""
        
        stats = {
            "strategies_used": list(self.plan_performance.keys()),
            "avg_performance_by_strategy": {}
        }
        
        for strategy, performances in self.plan_performance.items():
            if performances:
                stats["avg_performance_by_strategy"][strategy] = sum(performances) / len(performances)
        
        return stats
    
    def suggest_plan_improvements(self, strategy: str) -> List[str]:
        """Suggère des améliorations pour un type de plan"""
        
        suggestions = []
        
        if strategy in self.plan_performance:
            avg_perf = sum(self.plan_performance[strategy]) / len(self.plan_performance[strategy])
            
            if avg_perf < 0.5:
                suggestions.append("Considérer l'ajout d'étapes de vérification")
                suggestions.append("Augmenter le nombre de documents recherchés")
                suggestions.append("Ajouter une étape d'auto-réflexion")
            elif avg_perf > 0.8:
                suggestions.append("Plan performant - maintenir la stratégie actuelle")
                suggestions.append("Possibilité d'optimiser pour la rapidité")
        
        return suggestions

class AgenticRAG:
    """Système RAG Agentique avec planification et outils multiples"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        # Initialisation du modèle d'embeddings
        print("🔄 Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Stockage des documents et index FAISS
        self.documents: List[Document] = []
        self.index = None
        self.dimension = 384
        
        # Composants agentiques
        self.memory = ConversationMemory()
        self.classifier = QueryClassifier(self)
        
        # Outils disponibles (enrichis avec capacités multimodales)
        self.tools = {
            "search": DocumentSearchTool(self),
            "summarize": SummarizationTool(self),
            "compare": ComparisonTool(self),
            "analyze": AnalysisTool(self),
            "reflect": SelfReflectionTool(self),
            "enrich_context": ContextEnrichmentTool(self),
            "analyze_image": ImageAnalysisTool(self),
            "extract_text": OCRTool(self),
            "multimodal_analysis": MultimodalAnalysisTool(self)
        }
        
        self.planner = AgentPlanner(self.tools)
        
        print("✅ Système RAG Agentique initialisé avec succès!")
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Teste la connectivité à l'API Gemini"""
        print("🧪 Test de connectivité API Gemini...")
        
        test_result = {
            "gemini_text": {"status": "unknown", "latency": 0, "error": None},
            "gemini_vision": {"status": "unknown", "latency": 0, "error": None}
        }
        
        # Test API Gemini Text
        try:
            start_time = time.time()
            response = self.query_gemini("Test de connectivité - réponds simplement 'OK'")
            end_time = time.time()
            
            if "❌" not in response:
                test_result["gemini_text"]["status"] = "success"
                test_result["gemini_text"]["latency"] = round(end_time - start_time, 2)
                print("✅ API Gemini Text: Connecté")
            else:
                test_result["gemini_text"]["status"] = "error"
                test_result["gemini_text"]["error"] = response
                print(f"❌ API Gemini Text: {response}")
                
        except Exception as e:
            test_result["gemini_text"]["status"] = "error"
            test_result["gemini_text"]["error"] = str(e)
            print(f"❌ API Gemini Text: Exception {e}")
        
        # Test API Gemini Vision avec une petite image de test
        try:
            # Création d'une petite image de test (1x1 pixel blanc)
            test_image = Image.new('RGB', (1, 1), color='white')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='PNG')
            test_image_data = img_buffer.getvalue()
            
            start_time = time.time()
            image_tool = ImageAnalysisTool(self)
            result = image_tool.execute(
                image_data=test_image_data,
                query="Test de connectivité - décris cette image"
            )
            end_time = time.time()
            
            if result.get("success", False):
                test_result["gemini_vision"]["status"] = "success"
                test_result["gemini_vision"]["latency"] = round(end_time - start_time, 2)
                print("✅ API Gemini Vision: Connecté")
            else:
                test_result["gemini_vision"]["status"] = "error"
                test_result["gemini_vision"]["error"] = result.get("error", "Unknown error")
                print(f"❌ API Gemini Vision: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            test_result["gemini_vision"]["status"] = "error"
            test_result["gemini_vision"]["error"] = str(e)
            print(f"❌ API Gemini Vision: Exception {e}")
        
        return test_result
    
    # Méthodes RAG de base (identiques au système original)
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrait le texte d'un fichier PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                return text
        except Exception as e:
            raise Exception(f"Erreur lors de l'extraction du PDF: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Divise le texte en chunks avec chevauchement"""
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def add_pdf(self, pdf_path: str, metadata: Dict[str, Any] = None) -> None:
        """Ajoute un PDF au système RAG"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
        
        print(f"📄 Traitement du PDF: {pdf_path}")
        
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        
        if metadata is None:
            metadata = {}
        metadata.update({
            'source': pdf_path,
            'filename': Path(pdf_path).name,
            'total_chunks': len(chunks)
        })
        
        print(f"🔄 Création des embeddings pour {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = i
            
            embedding = self.embedding_model.encode(chunk)
            
            doc = Document(
                content=chunk,
                metadata=chunk_metadata,
                embedding=embedding
            )
            
            self.documents.append(doc)
            
            if (i + 1) % 20 == 0:
                print(f"   Progression: {i + 1}/{len(chunks)} chunks traités")
        
        self._build_index()
        print(f"✅ PDF ajouté avec succès: {len(chunks)} chunks indexés")
    
    def add_image(self, image_path: str, metadata: Dict[str, Any] = None) -> None:
        """Ajoute une image au système RAG avec analyse automatique"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Le fichier image {image_path} n'existe pas")
        
        print(f"🖼️  Traitement de l'image: {image_path}")
        
        # Lecture de l'image
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
        
        # Métadonnées par défaut
        if metadata is None:
            metadata = {}
        metadata.update({
            'source': image_path,
            'filename': Path(image_path).name,
            'file_type': 'image',
            'doc_type': 'image'
        })
        
        print("🔄 Analyse de l'image avec Gemini Vision...")
        
        # Analyse de l'image avec Gemini Vision
        image_tool = ImageAnalysisTool(self)
        analysis_result = image_tool.execute(
            image_data=image_data,
            query="Décris cette image en détail, en incluant tous les éléments visuels, texte, objets, personnes, et contexte visible."
        )
        
        if analysis_result.get("success", False):
            image_description = analysis_result["description"]
        else:
            image_description = f"Erreur lors de l'analyse: {analysis_result.get('error', 'Inconnue')}"
        
        print("🔄 Extraction du texte (OCR)...")
        
        # Extraction du texte via OCR
        ocr_tool = OCRTool(self)
        ocr_result = ocr_tool.execute(image_data=image_data)
        
        extracted_text = ""
        if ocr_result.get("success", False) and ocr_result.get("has_text", False):
            extracted_text = ocr_result["extracted_text"]
            print(f"   Texte extrait: {len(extracted_text)} caractères")
        else:
            print("   Aucun texte détecté dans l'image")
        
        # Combinaison description + texte extrait pour le contenu
        content_parts = [f"Description de l'image: {image_description}"]
        if extracted_text:
            content_parts.append(f"Texte extrait: {extracted_text}")
        
        content = "\n\n".join(content_parts)
        
        # Création de l'embedding
        print("🔄 Création de l'embedding...")
        embedding = self.embedding_model.encode(content)
        
        # Mise à jour des métadonnées
        metadata.update({
            'has_text': bool(extracted_text),
            'text_length': len(extracted_text),
            'description_length': len(image_description),
            'analysis_success': analysis_result.get("success", False)
        })
        
        # Création du document image
        doc = Document(
            content=content,
            metadata=metadata,
            embedding=embedding,
            doc_type="image",
            image_data=image_data,
            image_description=image_description
        )
        
        self.documents.append(doc)
        
        # Reconstruction de l'index FAISS
        self._build_index()
        print(f"✅ Image ajoutée avec succès: {Path(image_path).name}")
    
    def add_images_from_directory(self, directory_path: str, 
                                 supported_formats: List[str] = None) -> None:
        """Ajoute toutes les images d'un répertoire"""
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Le répertoire {directory_path} n'existe pas")
        
        image_files = []
        for ext in supported_formats:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"⚠️ Aucune image trouvée dans {directory_path}")
            return
        
        print(f"📁 Traitement de {len(image_files)} images du répertoire {directory_path}")
        
        for i, image_file in enumerate(image_files):
            try:
                print(f"\n📷 Image {i+1}/{len(image_files)}: {image_file.name}")
                self.add_image(str(image_file))
            except Exception as e:
                print(f"❌ Erreur avec {image_file.name}: {str(e)}")
        
        print(f"\n✅ Traitement terminé: {len(image_files)} images traitées")
    
    def search_images(self, query: str, k: int = 5) -> List[Document]:
        """Recherche spécifiquement dans les documents images"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Filtrage des documents images uniquement
        image_docs = [doc for doc in self.documents if doc.doc_type == "image"]
        
        if not image_docs:
            return []
        
        # Recherche dans les embeddings des images
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Construction d'un index temporaire pour les images seulement
        image_embeddings = np.array([doc.embedding for doc in image_docs])
        faiss.normalize_L2(image_embeddings)
        
        temp_index = faiss.IndexFlatIP(self.dimension)
        temp_index.add(image_embeddings.astype('float32'))
        
        # Recherche
        scores, indices = temp_index.search(query_embedding.astype('float32'), min(k, len(image_docs)))
        
        # Retour des documents trouvés
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                doc = image_docs[idx]
                doc_copy = Document(
                    content=doc.content,
                    metadata={**doc.metadata, 'similarity_score': float(score)},
                    embedding=doc.embedding,
                    doc_type=doc.doc_type,
                    image_data=doc.image_data,
                    image_description=doc.image_description
                )
                results.append(doc_copy)
        
        return results
    
    def _build_index(self) -> None:
        """Construit l'index FAISS avec tous les documents"""
        if not self.documents:
            return
        
        self.index = faiss.IndexFlatIP(self.dimension)
        embeddings = np.array([doc.embedding for doc in self.documents])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        print(f"🔍 Index FAISS construit avec {len(self.documents)} documents")
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Recherche les documents les plus similaires à la requête"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                doc = self.documents[idx]
                doc_copy = Document(
                    content=doc.content,
                    metadata={**doc.metadata, 'similarity_score': float(score)},
                    embedding=doc.embedding
                )
                results.append(doc_copy)
        
        return results
    
    def query_gemini(self, prompt: str, context: str = "") -> str:
        """Envoie une requête à l'API Gemini avec retry et meilleure gestion d'erreurs"""
        if context:
            full_prompt = f"""Contexte pertinent extrait des documents:
{context}

Question: {prompt}

Instructions:
- Réponds en te basant principalement sur le contexte fourni
- Si l'information n'est pas dans le contexte, indique-le clairement
- Sois précis et détaillé dans ta réponse
- Utilise un ton naturel et professionnel"""
        else:
            full_prompt = prompt
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        # Retry logic avec backoff exponentiel
        max_retries = 3
        base_timeout = 20
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (1.5 ** attempt)  # 20s, 30s, 45s
                
                response = requests.post(
                    f"{self.base_url}?key={self.api_key}",
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
                
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Erreur: Aucune réponse générée par Gemini"
                    
            except requests.exceptions.Timeout as e:
                print(f"⏱️ Timeout Gemini à la tentative {attempt + 1}: {timeout}s dépassé")
                if attempt == max_retries - 1:
                    return f"❌ Timeout API Gemini après {max_retries} tentatives. Réessayez plus tard."
                time.sleep(1 + attempt)
                
            except requests.exceptions.ConnectionError as e:
                print(f"🔌 Erreur de connexion Gemini à la tentative {attempt + 1}")
                if attempt == max_retries - 1:
                    return f"❌ Impossible de contacter l'API Gemini après {max_retries} tentatives. Vérifiez votre connexion."
                time.sleep(2 + attempt)
                
            except requests.exceptions.RequestException as e:
                print(f"🚫 Erreur de requête Gemini: {e}")
                if attempt == max_retries - 1:
                    return f"❌ Erreur API Gemini après {max_retries} tentatives: {str(e)}"
                time.sleep(1)
                
            except (KeyError, IndexError) as e:
                return f"❌ Format de réponse API inattendu: {str(e)}"
        
        return "❌ Échec de toutes les tentatives de requête Gemini"
    
    def execute_plan(self, plan: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Exécute un plan étape par étape avec monitoring et adaptation"""
        results = {}
        documents = []
        enriched_query = query
        execution_success = True
        
        print("🎯 Exécution du plan agentique...")
        
        for i, step in enumerate(plan):
            action = step["action"]
            params = step["params"].copy()
            
            try:
                print(f"   Étape {i+1}: {action}")
                
                # Enregistrement de l'action
                action_record = AgentAction(
                    action_type=action,
                    timestamp=datetime.now(),
                    confidence=step.get("confidence", 0.8),
                    reasoning=step.get("reasoning", f"Exécution de {action}")
                )
                
                if action == "search":
                    # Utiliser la requête enrichie si disponible
                    search_query = enriched_query if enriched_query != query else query
                    params["query"] = search_query
                    
                    search_result = self.tools["search"].execute(**params)
                    documents = search_result["results"]
                    results["search"] = search_result
                    
                    action_record.success = len(documents) > 0
                    action_record.feedback_score = min(len(documents) / params.get("k", 5), 1.0)
                
                elif action == "enrich_context":
                    if "enrich_context" in self.tools:
                        params["memory"] = self.memory
                        enrichment_result = self.tools["enrich_context"].execute(**params)
                        if enrichment_result.get("enrichment_applied", False):
                            enriched_query = enrichment_result["enriched_query"]
                        results["context_enrichment"] = enrichment_result
                        
                        action_record.success = enrichment_result.get("enrichment_applied", False)
                        action_record.feedback_score = 0.8 if action_record.success else 0.3
                
                elif action == "summarize":
                    if documents:
                        params["documents"] = documents
                        summary_result = self.tools["summarize"].execute(**params)
                        results["summary"] = summary_result
                        
                        action_record.success = True
                        action_record.feedback_score = 0.8  # Score par défaut
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                elif action == "compare":
                    if documents:
                        params["documents"] = documents
                        compare_result = self.tools["compare"].execute(**params)
                        results["comparison"] = compare_result
                        
                        action_record.success = True
                        action_record.feedback_score = 0.8
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                elif action == "analyze":
                    if documents:
                        params["documents"] = documents
                        analysis_result = self.tools["analyze"].execute(**params)
                        results["analysis"] = analysis_result
                        
                        action_record.success = True
                        action_record.feedback_score = 0.8
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                elif action == "reflect":
                    # Auto-réflexion sur la qualité de la réponse
                    if "generation" in results or "summary" in results or "analysis" in results:
                        # Prendre la dernière réponse générée
                        last_answer = ""
                        context_used = ""
                        
                        if "generation" in results:
                            last_answer = results["generation"]["answer"]
                            context_used = results["generation"].get("context_used", "")
                        elif "summary" in results:
                            last_answer = results["summary"]["summary"]
                        elif "analysis" in results:
                            last_answer = results["analysis"]["analysis"]
                        
                        if last_answer:
                            params["answer"] = last_answer
                            params["context"] = context_used
                            reflection_result = self.tools["reflect"].execute(**params)
                            results["reflection"] = reflection_result
                            
                            # Utilisation du score de qualité pour l'apprentissage
                            quality_score = reflection_result.get("quality_score", 7.0)
                            action_record.success = quality_score >= 6.0
                            action_record.feedback_score = quality_score / 10.0
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                elif action == "search_images":
                    # Recherche spécifique dans les images
                    search_query = enriched_query if enriched_query != query else query
                    params["query"] = search_query
                    
                    image_docs = self.search_images(**params)
                    results["image_search"] = {
                        "results": image_docs,
                        "count": len(image_docs)
                    }
                    
                    action_record.success = len(image_docs) > 0
                    action_record.feedback_score = min(len(image_docs) / params.get("k", 5), 1.0)
                
                elif action == "analyze_image":
                    # Analyse d'images trouvées
                    image_docs = results.get("image_search", {}).get("results", [])
                    if image_docs:
                        # Analyse de la première image la plus pertinente
                        best_image = image_docs[0]
                        if best_image.image_data:
                            analysis_result = self.tools["analyze_image"].execute(
                                image_data=best_image.image_data,
                                query=enriched_query
                            )
                            results["image_analysis"] = analysis_result
                            
                            action_record.success = analysis_result.get("success", False)
                            action_record.feedback_score = 0.8 if action_record.success else 0.2
                        else:
                            action_record.success = False
                            action_record.feedback_score = 0.0
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                elif action == "extract_text":
                    # Extraction de texte des images
                    image_docs = results.get("image_search", {}).get("results", [])
                    if image_docs:
                        extracted_texts = []
                        for img_doc in image_docs[:3]:  # Limite à 3 images
                            if img_doc.image_data:
                                ocr_result = self.tools["extract_text"].execute(
                                    image_data=img_doc.image_data
                                )
                                if ocr_result.get("success", False) and ocr_result.get("has_text", False):
                                    extracted_texts.append({
                                        "filename": img_doc.metadata.get("filename", "Unknown"),
                                        "text": ocr_result["extracted_text"]
                                    })
                        
                        results["text_extraction"] = {
                            "extracted_texts": extracted_texts,
                            "total_images_processed": len(image_docs),
                            "images_with_text": len(extracted_texts)
                        }
                        
                        action_record.success = len(extracted_texts) > 0
                        action_record.feedback_score = len(extracted_texts) / len(image_docs) if image_docs else 0.0
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                elif action == "multimodal_analysis":
                    # Analyse multimodale combinant texte et images
                    text_docs = documents  # Documents textuels de la recherche précédente
                    image_docs = results.get("image_search", {}).get("results", [])
                    
                    if text_docs or image_docs:
                        multimodal_result = self.tools["multimodal_analysis"].execute(
                            query=enriched_query,
                            text_documents=text_docs,
                            image_documents=image_docs
                        )
                        results["multimodal_analysis"] = multimodal_result
                        
                        action_record.success = multimodal_result.get("success", False)
                        action_record.feedback_score = 0.9 if action_record.success else 0.3
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                elif action == "generate":
                    # Génération standard avec contexte
                    if documents:
                        context_parts = [f"[Document {i+1}] {doc.content}" for i, doc in enumerate(documents)]
                        context = "\n\n".join(context_parts)
                        answer = self.query_gemini(enriched_query, context)
                        results["generation"] = {
                            "answer": answer,
                            "context_used": context,
                            "query_used": enriched_query
                        }
                        
                        action_record.success = len(answer) > 50  # Réponse substantielle
                        action_record.feedback_score = min(len(answer) / 200, 1.0)
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                    # Génération standard avec contexte
                    if documents:
                        context_parts = [f"[Document {i+1}] {doc.content}" for i, doc in enumerate(documents)]
                        context = "\n\n".join(context_parts)
                        answer = self.query_gemini(enriched_query, context)
                        results["generation"] = {
                            "answer": answer,
                            "context_used": context,
                            "query_used": enriched_query
                        }
                        
                        action_record.success = len(answer) > 50  # Réponse substantielle
                        action_record.feedback_score = min(len(answer) / 200, 1.0)
                    else:
                        action_record.success = False
                        action_record.feedback_score = 0.0
                
                # Enregistrement de l'action dans la mémoire
                self.memory.add_action(action_record)
                
                if not action_record.success:
                    execution_success = False
                    print(f"      ⚠️ Échec de l'étape {action}")
                
            except Exception as e:
                print(f"      ❌ Erreur lors de l'exécution de {action}: {str(e)}")
                action_record.success = False
                action_record.feedback_score = 0.0
                self.memory.add_action(action_record)
                execution_success = False
        
        # Calcul du score global d'exécution
        execution_score = sum(action.feedback_score for action in self.memory.action_history[-len(plan):]) / len(plan)
        
        # Enregistrement de la performance du plan
        if hasattr(self, 'planner'):
            strategy = getattr(self, '_last_strategy', 'unknown')
            self.planner.record_plan_performance(strategy, execution_score)
        
        results["execution_metadata"] = {
            "success": execution_success,
            "execution_score": execution_score,
            "enriched_query": enriched_query,
            "plan_length": len(plan)
        }
        
        return results
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Interface principale agentique pour poser des questions avec auto-amélioration"""
        print(f"🤔 Question: {question}")
        
        # Sauvegarde de la question dans la mémoire
        self.memory.add_message("user", question)
        
        # Classification intelligente de la requête
        classification = self.classifier.classify_query(question, self.memory)
        print(f"🏷️  Type détecté: {classification['type'].value} (confiance: {classification['confidence']:.2f})")
        print(f"📝 Raisonnement: {classification.get('reasoning', 'N/A')}")
        
        # Stockage de la stratégie pour l'apprentissage
        self._last_strategy = classification["strategy"]
        
        # Création du plan adaptatif
        plan = self.planner.create_plan(question, classification, self.memory)
        print(f"📋 Plan créé avec {len(plan)} étapes")
        
        # Affichage du plan détaillé
        for i, step in enumerate(plan):
            priority = step.get("priority", 1)
            reasoning = step.get("reasoning", "")
            print(f"   Étape {i+1}: {step['action']} (priorité: {priority:.1f}) {reasoning}")
        
        # Exécution du plan avec monitoring
        start_time = time.time()
        results = self.execute_plan(plan, question)
        end_time = time.time()
        
        # Construction de la réponse finale
        final_answer = self._build_final_response(results, question, classification)
        
        # Auto-évaluation et apprentissage
        self._perform_self_evaluation(question, final_answer, results, classification)
        
        # Sauvegarde de la réponse dans la mémoire avec métadonnées enrichies
        response_metadata = {
            "classification": classification,
            "execution_time": end_time - start_time,
            "tools_used": list(results.keys()),
            "execution_score": results.get("execution_metadata", {}).get("execution_score", 0.5),
            "plan_steps": len(plan)
        }
        
        self.memory.add_message("assistant", final_answer["answer"], response_metadata)
        
        # Suggestions d'amélioration si pertinentes
        if results.get("execution_metadata", {}).get("execution_score", 1.0) < 0.6:
            suggestions = self.planner.suggest_plan_improvements(classification["strategy"])
            if suggestions:
                final_answer["improvement_suggestions"] = suggestions
        
        return final_answer
    
    def _perform_self_evaluation(self, question: str, answer: Dict[str, Any], 
                               results: Dict[str, Any], classification: Dict[str, Any]):
        """Effectue une auto-évaluation pour l'apprentissage continu"""
        
        # Auto-réflexion si pas déjà effectuée
        if "reflection" not in results and "reflect" in self.tools:
            try:
                reflection_result = self.tools["reflect"].execute(
                    query=question,
                    answer=answer["answer"],
                    context=answer.get("sources", [])
                )
                
                # Apprentissage basé sur la réflexion
                quality_score = reflection_result.get("quality_score", 7.0)
                needs_improvement = reflection_result.get("needs_improvement", False)
                
                if needs_improvement:
                    print(f"🔍 Auto-évaluation: Score {quality_score}/10 - Amélioration nécessaire")
                    
                    # Ajustement des poids de classification si faible qualité
                    if quality_score < 5.0:
                        query_type = classification["type"].value
                        self.classifier.pattern_weights[query_type] *= 0.9  # Réduction du poids
                
            except Exception as e:
                logger.warning(f"Erreur lors de l'auto-évaluation: {e}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Retourne des insights sur l'apprentissage et les performances"""
        
        insights = {
            "classification_stats": self.classifier.get_classification_stats(),
            "planning_stats": self.planner.get_planning_stats(),
            "memory_stats": {
                "total_messages": len(self.memory.messages),
                "total_actions": len(self.memory.action_history),
                "performance_by_action": {}
            }
        }
        
        # Calcul des performances par type d'action
        action_performance = defaultdict(list)
        for action in self.memory.action_history:
            if action.feedback_score is not None:
                action_performance[action.action_type].append(action.feedback_score)
        
        for action_type, scores in action_performance.items():
            insights["memory_stats"]["performance_by_action"][action_type] = {
                "average_score": sum(scores) / len(scores),
                "total_executions": len(scores),
                "success_rate": sum(1 for score in scores if score > 0.5) / len(scores)
            }
        
        return insights
    
    def suggest_conversation_improvements(self) -> List[str]:
        """Suggère des améliorations basées sur l'analyse de la conversation"""
        
        suggestions = []
        
        if len(self.memory.messages) > 5:
            # Analyse des patterns de conversation
            user_messages = [msg for msg in self.memory.messages if msg["role"] == "user"]
            
            # Détection de questions répétitives
            recent_questions = [msg["content"] for msg in user_messages[-3:]]
            if len(set(recent_questions)) < len(recent_questions):
                suggestions.append("Détection de questions similaires - considérer la reformulation pour plus de précision")
            
            # Analyse de la longueur moyenne des réponses
            assistant_messages = [msg for msg in self.memory.messages if msg["role"] == "assistant"]
            if assistant_messages:
                avg_length = sum(len(msg["content"]) for msg in assistant_messages) / len(assistant_messages)
                if avg_length < 100:
                    suggestions.append("Réponses courtes détectées - considérer des questions plus spécifiques")
                elif avg_length > 1000:
                    suggestions.append("Réponses longues détectées - considérer la demande de résumés")
        
        # Suggestions basées sur les performances
        learning_insights = self.get_learning_insights()
        for action_type, stats in learning_insights["memory_stats"]["performance_by_action"].items():
            if stats["success_rate"] < 0.5:
                suggestions.append(f"Performance faible pour {action_type} - révision de la stratégie recommandée")
        
        return suggestions
    
    def _build_final_response(self, results: Dict[str, Any], question: str, 
                            classification: Dict[str, Any]) -> Dict[str, Any]:
        """Construit la réponse finale à partir des résultats (avec support multimodal)"""
        
        # Priorité aux analyses multimodales
        if "multimodal_analysis" in results:
            multimodal_result = results["multimodal_analysis"]
            if multimodal_result.get("success", False):
                return {
                    "answer": multimodal_result["synthesis"],
                    "type": "multimodal",
                    "sources": self._extract_multimodal_sources(results),
                    "execution_details": results
                }
        
        # Analyses d'images spécifiques
        if "image_analysis" in results:
            image_result = results["image_analysis"]
            if image_result.get("success", False):
                return {
                    "answer": image_result["description"],
                    "type": "image_analysis",
                    "sources": self._extract_image_sources(results),
                    "execution_details": results
                }
        
        # Extraction de texte OCR
        if "text_extraction" in results:
            extraction_result = results["text_extraction"]
            if extraction_result.get("images_with_text", 0) > 0:
                extracted_texts = extraction_result["extracted_texts"]
                answer = "Texte extrait des images:\n\n"
                for i, text_data in enumerate(extracted_texts):
                    answer += f"Image {i+1} ({text_data['filename']}):\n{text_data['text']}\n\n"
                
                return {
                    "answer": answer,
                    "type": "ocr_extraction",
                    "sources": self._extract_image_sources(results),
                    "execution_details": results
                }
        
        # Résumés, comparaisons, analyses (logique existante)
        if "summary" in results:
            return {
                "answer": results["summary"]["summary"],
                "type": "summary",
                "sources": self._extract_sources(results.get("search", {}).get("results", [])),
                "execution_details": results
            }
        
        elif "comparison" in results:
            return {
                "answer": results["comparison"]["comparison"],
                "type": "comparison",
                "sources": self._extract_sources(results.get("search", {}).get("results", [])),
                "execution_details": results
            }
        
        elif "analysis" in results:
            return {
                "answer": results["analysis"]["analysis"],
                "type": "analysis",
                "sources": self._extract_sources(results.get("search", {}).get("results", [])),
                "execution_details": results
            }
        
        elif "generation" in results:
            return {
                "answer": results["generation"]["answer"],
                "type": "standard",
                "sources": self._extract_sources(results.get("search", {}).get("results", [])),
                "execution_details": results
            }
        
        else:
            return {
                "answer": "❌ Aucun résultat n'a pu être généré.",
                "type": "error",
                "sources": [],
                "execution_details": results
            }
    
    def _extract_multimodal_sources(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait les sources multimodales (texte + images)"""
        sources = []
        
        # Sources textuelles
        text_sources = self._extract_sources(results.get("search", {}).get("results", []))
        for source in text_sources:
            source["source_type"] = "text"
            sources.append(source)
        
        # Sources images
        image_sources = self._extract_image_sources(results)
        for source in image_sources:
            source["source_type"] = "image"
            sources.append(source)
        
        return sources
    
    def _extract_image_sources(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait les informations de source des images"""
        sources = []
        image_docs = results.get("image_search", {}).get("results", [])
        
        for doc in image_docs:
            sources.append({
                'filename': doc.metadata.get('filename', 'Unknown'),
                'similarity_score': doc.metadata.get('similarity_score', 0.0),
                'doc_type': 'image',
                'has_text': doc.metadata.get('has_text', False),
                'analysis_success': doc.metadata.get('analysis_success', False)
            })
        
        return sources
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extrait les informations de source des documents"""
        sources = []
        for doc in documents:
            sources.append({
                'filename': doc.metadata.get('filename', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 0),
                'similarity_score': doc.metadata.get('similarity_score', 0.0)
            })
        return sources
    
    # Méthodes utilitaires (identiques au système original)
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système (avec support multimodal)"""
        
        # Séparation des documents par type
        text_docs = [doc for doc in self.documents if doc.doc_type == "text"]
        image_docs = [doc for doc in self.documents if doc.doc_type == "image"]
        
        # Statistiques des images
        images_with_text = sum(1 for doc in image_docs if doc.metadata.get('has_text', False))
        successful_analyses = sum(1 for doc in image_docs if doc.metadata.get('analysis_success', False))
        
        return {
            'total_documents': len(self.documents),
            'text_documents': len(text_docs),
            'image_documents': len(image_docs),
            'images_with_text': images_with_text,
            'successful_image_analyses': successful_analyses,
            'index_built': self.index is not None,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name,
            'tools_available': list(self.tools.keys()),
            'conversation_length': len(self.memory.messages),
            'multimodal_capable': True
        }
    
    def save_index(self, filepath: str) -> None:
        """Sauvegarde l'index et les documents"""
        data = {
            'documents': [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'embedding': doc.embedding.tolist()
                }
                for doc in self.documents
            ],
            'memory': {
                'messages': self.memory.messages,
                'context': self.memory.context
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        if self.index is not None:
            faiss.write_index(self.index, f"{filepath}.faiss")
        
        print(f"💾 Index et mémoire sauvegardés dans {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """Charge l'index et les documents"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = []
        for doc_data in data['documents']:
            doc = Document(
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                embedding=np.array(doc_data['embedding'])
            )
            self.documents.append(doc)
        
        # Chargement de la mémoire si disponible
        if 'memory' in data:
            self.memory.messages = data['memory'].get('messages', [])
            self.memory.context = data['memory'].get('context', {})
        
        # Chargement de l'index FAISS
        faiss_path = f"{filepath}.faiss"
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        else:
            self._build_index()
        
        print(f"📂 Index et mémoire chargés depuis {filepath} avec {len(self.documents)} documents")

def interactive_agentic_chat(rag_system: AgenticRAG):
    """Interface de chat interactive pour le système agentique avancé"""
    print("\n" + "="*80)
    print("🤖 SYSTÈME RAG AGENTIQUE AVANCÉ - CHAT INTERACTIF")
    print("="*80)
    print("💡 Commandes disponibles:")
    print("   • Tapez votre question pour interroger les documents")
    print("   • 'memory' - Afficher l'historique de conversation")
    print("   • 'stats' - Afficher les statistiques système")
    print("   • 'insights' - Afficher les insights d'apprentissage")
    print("   • 'tools' - Lister les outils disponibles")
    print("   • 'suggestions' - Obtenir des suggestions d'amélioration")
    print("   • 'clear' - Effacer la mémoire de conversation")
    print("   • 'debug' - Mode debug avec détails d'exécution")
    print("   • 'help' - Afficher cette aide")
    print("   • 'quit' ou 'exit' - Quitter")
    print("="*80)
    
    debug_mode = False
    
    while True:
        try:
            user_input = input("\n🔤 Votre question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Au revoir!")
                break
            
            elif user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"🔧 Mode debug: {'ACTIVÉ' if debug_mode else 'DÉSACTIVÉ'}")
                continue
            
            elif user_input.lower() == 'memory':
                print("\n🧠 MÉMOIRE DE CONVERSATION:")
                recent_context = rag_system.memory.get_recent_context(10)
                if recent_context:
                    print(recent_context)
                else:
                    print("   Aucune conversation précédente")
                
                # Affichage des actions récentes
                if rag_system.memory.action_history:
                    print(f"\n🎬 ACTIONS RÉCENTES ({len(rag_system.memory.action_history[-5:])}):")
                    for action in rag_system.memory.action_history[-5:]:
                        success_icon = "✅" if action.success else "❌"
                        score = action.feedback_score or 0.0
                        print(f"   {success_icon} {action.action_type} (score: {score:.2f}) - {action.reasoning}")
                continue
            
            elif user_input.lower() == 'insights':
                print("\n🧠 INSIGHTS D'APPRENTISSAGE:")
                insights = rag_system.get_learning_insights()
                
                # Classification
                class_stats = insights.get("classification_stats", {})
                if "average_confidence" in class_stats:
                    print(f"   📊 Confiance moyenne de classification: {class_stats['average_confidence']:.2f}")
                if "type_distribution" in class_stats:
                    print("   📈 Distribution des types de requêtes:")
                    for qtype, count in class_stats["type_distribution"].items():
                        print(f"      • {qtype}: {count}")
                
                # Performance des actions
                action_stats = insights["memory_stats"]["performance_by_action"]
                if action_stats:
                    print("\n   🎯 Performance par type d'action:")
                    for action_type, stats in action_stats.items():
                        success_rate = stats["success_rate"] * 100
                        avg_score = stats["average_score"]
                        print(f"      • {action_type}: {success_rate:.1f}% succès, score moy: {avg_score:.2f}")
                
                continue
            
            elif user_input.lower() == 'suggestions':
                print("\n💡 SUGGESTIONS D'AMÉLIORATION:")
                suggestions = rag_system.suggest_conversation_improvements()
                if suggestions:
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {i}. {suggestion}")
                else:
                    print("   Aucune suggestion disponible - performance satisfaisante")
                continue
            
            elif user_input.lower() == 'stats':
                stats = rag_system.get_stats()
                print("\n📊 STATISTIQUES DU SYSTÈME:")
                print(f"   📄 Documents indexés: {stats['total_documents']}")
                print(f"   🔍 Index construit: {'✅ Oui' if stats['index_built'] else '❌ Non'}")
                print(f"   🎯 Dimension embeddings: {stats['embedding_dimension']}")
                print(f"   🤖 Modèle: {stats['model_name']}")
                print(f"   🛠️  Outils disponibles: {', '.join(stats['tools_available'])}")
                print(f"   💬 Messages en mémoire: {stats['conversation_length']}")
                
                # Statistiques de planification
                planning_stats = rag_system.planner.get_planning_stats()
                if planning_stats.get("avg_performance_by_strategy"):
                    print("\n   📋 Performance des stratégies:")
                    for strategy, performance in planning_stats["avg_performance_by_strategy"].items():
                        print(f"      • {strategy}: {performance:.2f}")
                
                continue
            
            elif user_input.lower() == 'tools':
                print("\n🛠️  OUTILS DISPONIBLES:")
                for name, tool in rag_system.tools.items():
                    print(f"   • {name}: {tool.get_description()}")
                
                # Performance des outils
                insights = rag_system.get_learning_insights()
                action_stats = insights["memory_stats"]["performance_by_action"]
                if action_stats:
                    print("\n   📈 Performance des outils:")
                    for tool_name in rag_system.tools.keys():
                        if tool_name in action_stats:
                            stats = action_stats[tool_name]
                            print(f"      • {tool_name}: {stats['success_rate']*100:.1f}% succès")
                
                continue
            
            elif user_input.lower() == 'clear':
                rag_system.memory = ConversationMemory()
                print("🧹 Mémoire de conversation effacée")
                continue
            
            elif user_input.lower() == 'help':
                print("\n💡 AIDE SYSTÈME AGENTIQUE AVANCÉ:")
                print("   Le système dispose maintenant de capacités d'apprentissage:")
                print("   • 🧠 Apprentissage adaptatif des patterns de questions")
                print("   • 🎯 Optimisation automatique des stratégies")
                print("   • 🔍 Auto-réflexion sur la qualité des réponses")
                print("   • 📊 Monitoring continu des performances")
                print("   • 🔧 Adaptation des plans basée sur l'historique")
                print("\n   Exemples de questions avancées:")
                print("     - 'Analyse en profondeur les implications de...'")
                print("     - 'Compare de manière critique les approches...'")
                print("     - 'Synthétise les points clés en tenant compte de...'")
                print("     - Questions de suivi basées sur le contexte")
                continue
            
            # Traitement de la question avec le système agentique avancé
            start_time = time.time()
            
            if debug_mode:
                print("\n🔧 MODE DEBUG ACTIVÉ")
                print("=" * 40)
            
            result = rag_system.ask(user_input)
            end_time = time.time()
            
            print("\n" + "="*80)
            print("🤖 RÉPONSE AGENTIQUE AVANCÉE:")
            print("="*80)
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📚 SOURCES UTILISÉES ({len(result['sources'])}):")
                for i, source in enumerate(result['sources']):
                    score_bar = "█" * int(source['similarity_score'] * 10)
                    print(f"   {i+1}. 📄 {source['filename']} (chunk {source['chunk_id']})")
                    print(f"      Pertinence: {score_bar} {source['similarity_score']:.3f}")
            
            # Affichage des détails d'exécution
            exec_details = result.get('execution_details', {})
            if exec_details:
                tools_used = list(exec_details.keys())
                print(f"\n🛠️  Outils utilisés: {', '.join(tools_used)}")
                
                execution_metadata = exec_details.get('execution_metadata', {})
                if execution_metadata:
                    exec_score = execution_metadata.get('execution_score', 0)
                    print(f"🎯 Score d'exécution: {exec_score:.2f}")
                    
                    if 'enriched_query' in execution_metadata:
                        enriched = execution_metadata['enriched_query']
                        if enriched != user_input:
                            print(f"🔍 Requête enrichie: {enriched}")
            
            # Suggestions d'amélioration si disponibles
            if 'improvement_suggestions' in result:
                print(f"\n💡 SUGGESTIONS D'AMÉLIORATION:")
                for suggestion in result['improvement_suggestions']:
                    print(f"   • {suggestion}")
            
            print(f"\n⏱️  Temps de réponse: {end_time - start_time:.2f}s")
            print(f"🏷️  Type de réponse: {result['type']}")
            
            # Informations de debug
            if debug_mode and 'execution_details' in result:
                print(f"\n🔧 DÉTAILS DEBUG:")
                exec_meta = result['execution_details'].get('execution_metadata', {})
                print(f"   Plan exécuté: {exec_meta.get('plan_length', 0)} étapes")
                print(f"   Succès global: {exec_meta.get('success', 'Unknown')}")
                
            print("="*80)
        
        except KeyboardInterrupt:
            print("\n\n👋 Arrêt du programme...")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {str(e)}")
            logger.error(f"Error in interactive chat: {e}", exc_info=True)

def main():
    """Fonction principale avec interface interactive agentique"""
    # Configuration - Get API key from environment
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
    if not API_KEY:
        print("❌ Erreur: GOOGLE_API_KEY non défini!")
        print("💡 Configurez votre clé API:")
        print("   PowerShell: $env:GOOGLE_API_KEY=\"votre_clé_ici\"")
        print("   Cmd: set GOOGLE_API_KEY=votre_clé_ici")
        print("   Ou créez un fichier .env avec: GOOGLE_API_KEY=votre_clé_ici")
        return
    
    print("🚀 SYSTÈME RAG AGENTIQUE AVEC GEMINI")
    print("="*60)
    
    # Initialisation du système RAG agentique
    try:
        rag = AgenticRAG(api_key=API_KEY)
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {str(e)}")
        return
    
    # Menu principal
    while True:
        print("\n📋 MENU PRINCIPAL:")
        print("1. 📄 Ajouter un PDF")
        print("2. 🖼️  Ajouter une image")
        print("3. 📁 Ajouter des images d'un dossier")
        print("4. 🤖 Chat agentique multimodal")
        print("5. 💾 Sauvegarder l'index et la mémoire")
        print("6. 📂 Charger un index")
        print("7. 📊 Afficher les statistiques")
        print("8. 🛠️  Tester les outils")
        print("9. 🔌 Tester la connectivité API")
        print("10. ❌ Quitter")
        
        choice = input("\n🔤 Votre choix (1-10): ").strip()
        
        if choice == '1':
            pdf_path = input("📁 Chemin vers le fichier PDF: ").strip()
            if pdf_path:
                try:
                    rag.add_pdf(pdf_path)
                except Exception as e:
                    print(f"❌ Erreur: {str(e)}")
        
        elif choice == '2':
            image_path = input("🖼️  Chemin vers l'image: ").strip()
            if image_path:
                try:
                    rag.add_image(image_path)
                except Exception as e:
                    print(f"❌ Erreur: {str(e)}")
        
        elif choice == '3':
            directory_path = input("📁 Chemin vers le dossier d'images: ").strip()
            if directory_path:
                try:
                    rag.add_images_from_directory(directory_path)
                except Exception as e:
                    print(f"❌ Erreur: {str(e)}")
        
        elif choice == '4':
            if len(rag.documents) == 0:
                print("⚠️  Aucun document chargé. Ajoutez d'abord des PDFs ou des images.")
            else:
                interactive_agentic_chat(rag)
        
        elif choice == '5':
            if len(rag.documents) > 0:
                filename = input("💾 Nom du fichier de sauvegarde (ex: mon_index.json): ").strip()
                if filename:
                    rag.save_index(filename)
            else:
                print("⚠️  Aucun document à sauvegarder.")
        
        elif choice == '6':
            filename = input("📂 Nom du fichier à charger: ").strip()
            if filename and os.path.exists(filename):
                try:
                    rag.load_index(filename)
                except Exception as e:
                    print(f"❌ Erreur: {str(e)}")
            else:
                print("❌ Fichier introuvable.")
        
        elif choice == '7':
            stats = rag.get_stats()
            print("\n📊 STATISTIQUES MULTIMODALES:")
            print(f"   📄 Documents totaux: {stats['total_documents']}")
            print(f"   📝 Documents texte: {stats['text_documents']}")
            print(f"   🖼️  Documents image: {stats['image_documents']}")
            print(f"   📖 Images avec texte: {stats['images_with_text']}")
            print(f"   ✅ Analyses d'images réussies: {stats['successful_image_analyses']}")
            print(f"   🔍 Index: {'✅ Construit' if stats['index_built'] else '❌ Non construit'}")
            print(f"   🎯 Dimension: {stats['embedding_dimension']}")
            print(f"   🤖 Modèle: {stats['model_name']}")
            print(f"   🛠️  Outils: {', '.join(stats['tools_available'])}")
            print(f"   💬 Conversation: {stats['conversation_length']} messages")
            print(f"   🌈 Capacité multimodale: {'✅ Oui' if stats['multimodal_capable'] else '❌ Non'}")
        
        elif choice == '8':
            print("\n🛠️  TEST DES OUTILS MULTIMODAUX:")
            for name, tool in rag.tools.items():
                print(f"   • {name}: {tool.get_description()}")
            
            if len(rag.documents) > 0:
                test_query = "test de fonctionnement multimodal"
                print(f"\n🧪 Test avec la requête: '{test_query}'")
                classification = rag.classifier.classify_query(test_query, rag.memory)
                print(f"   Classification: {classification}")
                plan = rag.planner.create_plan(test_query, classification, rag.memory)
                print(f"   Plan généré: {plan}")
                
                # Test spécifique pour les images
                image_docs = [doc for doc in rag.documents if doc.doc_type == "image"]
                if image_docs:
                    print(f"   📷 Images disponibles: {len(image_docs)}")
                    test_image_search = rag.search_images("test", k=2)
                    print(f"   🔍 Test recherche image: {len(test_image_search)} résultats")
            else:
                print("   ⚠️  Ajoutez des documents pour tester les outils")
            
            # Test simple de connectivité pour les outils d'image
            print("\n🧪 Test rapide de connectivité:")
            try:
                simple_test = rag.query_gemini("Test simple - réponds 'OK'")
                if "❌" not in simple_test:
                    print("   ✅ API Gemini Text: Fonctionnelle")
                else:
                    print(f"   ❌ API Gemini Text: {simple_test}")
            except Exception as e:
                print(f"   ❌ API Gemini Text: Exception {e}")
        
        elif choice == '9':
            # Test de connectivité API
            print("\n🔌 TEST DE CONNECTIVITÉ API:")
            test_results = rag.test_api_connection()
            
            print("\n📊 RÉSULTATS:")
            for api_name, result in test_results.items():
                status_icon = "✅" if result["status"] == "success" else "❌"
                api_display = "Gemini Text" if api_name == "gemini_text" else "Gemini Vision"
                print(f"   {status_icon} {api_display}: {result['status']}")
                
                if result["status"] == "success":
                    print(f"      ⏱️ Latence: {result['latency']}s")
                elif result["error"]:
                    print(f"      � Erreur: {result['error'][:100]}...")
            
            # Suggestions basées sur les résultats
            if all(r["status"] == "error" for r in test_results.values()):
                print("\n💡 SUGGESTIONS:")
                print("   • Vérifiez votre connexion internet")
                print("   • Vérifiez que votre clé API Gemini est valide")
                print("   • Essayez de redémarrer le programme")
                print("   • Consultez les logs d'erreur ci-dessus")
        
        elif choice == '10':
            print("�👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide. Veuillez entrer un numéro entre 1 et 10.")

if __name__ == "__main__":
    main()

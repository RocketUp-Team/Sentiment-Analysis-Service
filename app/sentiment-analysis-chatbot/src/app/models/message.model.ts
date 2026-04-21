export interface AspectSentiment {
  aspect: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | 'LOADING' | 'ERROR' | null;
  confidenceScore?: number;
  aspects?: AspectSentiment[];
  sarcasm_flag?: boolean;
  latency_ms?: number;
  // SHAP explanation (populated on demand via /explain)
  explainData?: ExplainResponse;
  explainLoading?: boolean;
}

export interface PredictRequest {
  text: string;
  lang?: string | null;
}

export interface ExplainRequest {
  text: string;
  lang?: string | null;
}

export interface ExplainResponse {
  tokens: string[];
  shap_values: number[];
  base_value: number;
  latency_ms: number;
}

export interface PredictResponse {
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  aspects: AspectSentiment[];
  sarcasm_flag: boolean;
  detected_lang: string;
  lang_confidence: number;
  latency_ms: number;
}

export interface BatchItemResult {
  row: number;
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  aspects: AspectSentiment[];
  error?: string;
}

export interface BatchPredictResponse {
  total_items: number;
  processed_items: number;
  failed_items: number;
  latency_ms: number;
  results: BatchItemResult[];
}


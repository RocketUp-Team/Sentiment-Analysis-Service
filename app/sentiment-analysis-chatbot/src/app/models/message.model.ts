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
}

export interface PredictRequest {
  text: string;
  lang: 'vi' | 'en';
}

export interface PredictResponse {
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  aspects: AspectSentiment[];
  sarcasm_flag: boolean;
  latency_ms: number;
}


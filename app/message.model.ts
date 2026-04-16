export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | 'LOADING' | 'ERROR' | null;
  confidenceScore?: number; // e.g., 0.9856
}
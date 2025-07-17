import React, { useState } from 'react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface PlanResponse {
  tips: string[];
  short_term: string[];
  medium_term: string[];
  long_term: string[];
  cost_estimate: string;
  time_estimate: string;
  employee_suggestion: string;
  additional_info?: any;
  chat: ChatMessage[];
}

const App: React.FC = () => {
  const [idea, setIdea] = useState('');
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!idea.trim()) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/startup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          idea: idea.trim(),
          history: plan?.chat || []
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }

      const data = await response.json();
      setPlan(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-blue-100 px-2">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-2xl p-8 border border-blue-100">
        <h1 className="text-3xl md:text-4xl font-extrabold text-blue-700 text-center mb-2 tracking-tight">Startup Assistant AI</h1>
        <p className="text-base md:text-lg text-gray-600 text-center mb-8">Share your startup idea and get a personalized plan!</p>

        {/* Chat history */}
        {plan && plan.chat && plan.chat.length > 0 && (
          <div className="mb-8 max-h-64 overflow-y-auto bg-blue-50 rounded-lg p-4 shadow-inner">
            <h3 className="text-blue-700 font-semibold mb-2 text-center">Conversation</h3>
            <ul className="space-y-3">
              {plan.chat.map((msg, idx) => (
                <li key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`px-4 py-2 rounded-lg max-w-[80%] text-sm ${msg.role === 'user' ? 'bg-blue-200 text-right text-blue-900' : 'bg-white border text-left text-gray-800'}`}>
                    <span className="block font-semibold mb-1">{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                    <span>{msg.content}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        <form onSubmit={handleSubmit} className="mb-8">
          <label htmlFor="idea" className="block text-sm font-medium text-gray-700 mb-2">
            Describe your startup idea:
          </label>
          <textarea
            id="idea"
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            placeholder="e.g., I want to build a mobile app that helps people find local farmers markets..."
            className="w-full h-28 md:h-32 p-3 border-2 border-blue-200 rounded-lg focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition outline-none resize-none bg-blue-50 placeholder-gray-400 text-gray-800"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !idea.trim()}
            className="mt-4 w-full bg-blue-600 text-white font-semibold py-2 rounded-lg shadow hover:bg-blue-700 transition disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            {loading ? 'Generating Plan...' : 'Get Startup Plan'}
          </button>
        </form>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 mb-4 rounded-lg text-center animate-pulse">
            {error}
          </div>
        )}

        {plan && (
          <div className="mt-6">
            <h2 className="text-xl font-bold text-blue-700 mb-4 text-center">Your Startup Plan</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-blue-50 rounded-lg p-4 shadow-sm">
                <h3 className="text-lg font-semibold mb-2 text-blue-600 flex items-center gap-2">💡 Tips</h3>
                <ul className="space-y-2 list-disc list-inside">
                  {plan.tips.map((tip, index) => (
                    <li key={index} className="text-gray-700">{tip}</li>
                  ))}
                </ul>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 shadow-sm">
                <h3 className="text-lg font-semibold mb-2 text-blue-600 flex items-center gap-2">💰 Cost Estimate</h3>
                <p className="text-gray-700 mb-2">{plan.cost_estimate}</p>
                <h3 className="text-lg font-semibold mb-2 text-blue-600 mt-4 flex items-center gap-2">⏱️ Time Estimate</h3>
                <p className="text-gray-700 mb-2">{plan.time_estimate}</p>
                <h3 className="text-lg font-semibold mb-2 text-blue-600 mt-4 flex items-center gap-2">👥 Team Size</h3>
                <p className="text-gray-700">{plan.employee_suggestion}</p>
              </div>
            </div>
            <div className="mt-8">
              <h3 className="text-lg font-semibold mb-3 text-blue-700">📅 Timeline</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-green-50 rounded-lg p-3 shadow-sm">
                  <h4 className="font-medium text-green-700 mb-2">Short Term</h4>
                  <ul className="space-y-1 list-disc list-inside">
                    {plan.short_term.map((item, index) => (
                      <li key={index} className="text-sm text-gray-700">{item}</li>
                    ))}
                  </ul>
                </div>
                <div className="bg-yellow-50 rounded-lg p-3 shadow-sm">
                  <h4 className="font-medium text-yellow-700 mb-2">Medium Term</h4>
                  <ul className="space-y-1 list-disc list-inside">
                    {plan.medium_term.map((item, index) => (
                      <li key={index} className="text-sm text-gray-700">{item}</li>
                    ))}
                  </ul>
                </div>
                <div className="bg-blue-50 rounded-lg p-3 shadow-sm">
                  <h4 className="font-medium text-blue-700 mb-2">Long Term</h4>
                  <ul className="space-y-1 list-disc list-inside">
                    {plan.long_term.map((item, index) => (
                      <li key={index} className="text-sm text-gray-700">{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
            {plan.additional_info?.note && (
              <div className="mt-6 text-center text-xs text-gray-400">{plan.additional_info.note}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App; 
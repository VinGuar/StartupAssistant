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
  follow_up_question?: string; // Add this field
}

// Add a list of all possible extra fields
const EXTRA_FIELDS = [
  { key: 'industry', label: 'Industry' },
  { key: 'team_size', label: 'Team Size' },
  { key: 'technical_skills', label: 'Technical Skills' },
  { key: 'monetization', label: 'Monetization Strategy' },
  { key: 'competition', label: 'Competition' },
  { key: 'unique_value', label: 'Unique Value Proposition' },
  { key: 'distribution', label: 'Distribution Channels' },
  { key: 'marketing', label: 'Marketing Approach' },
  { key: 'goals', label: 'Goals' },
  { key: 'challenges', label: 'Challenges' },
  { key: 'resources', label: 'Resources Available' },
  { key: 'legal', label: 'Legal/Regulatory' },
  { key: 'partners', label: 'Potential Partners' },
  { key: 'scalability', label: 'Scalability' },
  { key: 'timeline', label: 'Desired Timeline' },
  { key: 'funding', label: 'Funding Sources' },
  { key: 'experience', label: 'Relevant Experience' },
  { key: 'technology', label: 'Technology Stack' },
  { key: 'other', label: 'Other' },
];

const BASE_FIELDS = [
  { key: 'idea', label: 'Startup Idea', placeholder: 'Describe your startup idea...' },
  { key: 'location', label: 'Location', placeholder: 'Where will your startup operate?' },
  { key: 'budget', label: 'Budget', placeholder: 'How much money do you have or need?' },
  { key: 'target', label: 'Target Audience', placeholder: 'Who is your target customer?' },
  { key: 'time', label: 'Time Available', placeholder: 'How much time can you commit?' },
];

const App: React.FC = () => {
  const [idea, setIdea] = useState('');
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [error, setError] = useState('');
  const [fields, setFields] = useState<{ [key: string]: string }>(() => BASE_FIELDS.reduce((acc, f) => ({ ...acc, [f.key]: '' }), {}));
  const [addedFields, setAddedFields] = useState<string[]>([]);

  const handleFieldChange = (key: string, value: string) => {
    setFields((prev) => ({ ...prev, [key]: value }));
  };

  const handleAddField = (key: string) => {
    setAddedFields((prev) => [...prev, key]);
    setFields((prev) => ({ ...prev, [key]: '' }));
  };

  const availableExtraFields = EXTRA_FIELDS.filter(f => !addedFields.includes(f.key) && !BASE_FIELDS.some(bf => bf.key === f.key));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // Only send fields with values
    const filledFields = Object.fromEntries(Object.entries(fields).filter(([_, v]) => v && v.trim() !== ''));
    if (!filledFields.idea) return;
    setLoading(true);
    setError('');
    // Build full chat history including the new user message (as a summary of all fields)
    const userMsg = Object.entries(filledFields).map(([k, v]) => `${k}: ${v}`).join('\n');
    const newChatHistory = [
      ...(plan?.chat || []),
      { role: 'user', content: userMsg }
    ];
    try {
      const response = await fetch('http://localhost:8000/api/startup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fields: filledFields,
          history: newChatHistory
        }),
      });
      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }
      const data = await response.json();
      setPlan(data);
      setFields(BASE_FIELDS.reduce((acc, f) => ({ ...acc, [f.key]: '' }), {}));
      setAddedFields([]);
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

        {/* If follow_up_question is present, show it and let user answer */}
        {plan && plan.follow_up_question ? (
          <form onSubmit={handleSubmit} className="mb-8">
            <label htmlFor="idea" className="block text-sm font-medium text-gray-700 mb-2">
              {plan.follow_up_question}
            </label>
            <textarea
              id="idea"
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="Type your answer here..."
              className="w-full h-28 md:h-32 p-3 border-2 border-blue-200 rounded-lg focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition outline-none resize-none bg-blue-50 placeholder-gray-400 text-gray-800"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !idea.trim()}
              className="mt-4 w-full bg-blue-600 text-white font-semibold py-2 rounded-lg shadow hover:bg-blue-700 transition disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </form>
        ) : (
          <form onSubmit={handleSubmit} className="mb-8">
            {BASE_FIELDS.map(f => (
              <div key={f.key} className="mb-4">
                <label htmlFor={f.key} className="block text-sm font-medium text-gray-700 mb-1">{f.label}</label>
                <input
                  id={f.key}
                  type="text"
                  value={fields[f.key] || ''}
                  onChange={e => handleFieldChange(f.key, e.target.value)}
                  placeholder={f.placeholder}
                  className="w-full p-2 border-2 border-blue-200 rounded-lg focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition outline-none bg-blue-50 placeholder-gray-400 text-gray-800"
                  disabled={loading}
                />
              </div>
            ))}
            {addedFields.map(key => {
              const field = EXTRA_FIELDS.find(f => f.key === key);
              if (!field) return null;
              return (
                <div key={key} className="mb-4">
                  <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-1">{field.label}</label>
                  <input
                    id={key}
                    type="text"
                    value={fields[key] || ''}
                    onChange={e => handleFieldChange(key, e.target.value)}
                    placeholder={field.label}
                    className="w-full p-2 border-2 border-blue-200 rounded-lg focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition outline-none bg-blue-50 placeholder-gray-400 text-gray-800"
                    disabled={loading}
                  />
                </div>
              );
            })}
            {availableExtraFields.length > 0 && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Add More Fields</label>
                <select
                  className="w-full p-2 border-2 border-blue-200 rounded-lg bg-blue-50 text-gray-800"
                  onChange={e => {
                    if (e.target.value) handleAddField(e.target.value);
                    e.target.value = '';
                  }}
                  defaultValue=""
                  disabled={loading}
                >
                  <option value="" disabled>Add a field...</option>
                  {availableExtraFields.map(f => (
                    <option key={f.key} value={f.key}>{f.label}</option>
                  ))}
                </select>
              </div>
            )}
            <button
              type="submit"
              disabled={loading || !fields.idea.trim()}
              className="mt-4 w-full bg-blue-600 text-white font-semibold py-2 rounded-lg shadow hover:bg-blue-700 transition disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating Plan...' : 'Get Startup Plan'}
            </button>
          </form>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 mb-4 rounded-lg text-center animate-pulse">
            {error}
          </div>
        )}

        {/* Only show the plan if there is no follow_up_question */}
        {plan && !plan.follow_up_question && (
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
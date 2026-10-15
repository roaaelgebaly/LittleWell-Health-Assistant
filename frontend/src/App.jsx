import { useState } from 'react'
import './App.css'
import heroImage from './assets/hero.png'

const API_URL = 'http://localhost:8000/chat'

const makeWelcomeMessage = () => ({
  id: 'welcome',
  role: 'assistant',
  kind: 'welcome',
  content: "How can I help you with your child's health today?",
})

const initialChatState = {
  mode: 'ready',
  threadId: null,
  messages: [makeWelcomeMessage()],
}

const confidenceMap = {
  high: 'High confidence',
  medium: 'Moderate confidence',
  low: 'Low confidence',
  insufficient: 'Insufficient evidence',
}

function App() {
  const [chatState, setChatState] = useState(initialChatState)
  const [draft, setDraft] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const sendToBackend = async (payload) => {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      throw new Error(`Backend request failed: ${response.status}`)
    }

    return response.json()
  }

  const handleUserQuestion = async (questionText) => {
    const question = questionText.trim()
    if (!question) return

    setIsLoading(true)

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      kind: 'user',
      content: question,
    }

    const payload =
      chatState.mode === 'awaiting_clarification'
        ? { user_reply: question, thread_id: chatState.threadId }
        : { question, thread_id: chatState.threadId }

    setChatState((prev) => ({
      ...prev,
      mode: 'pending',
      messages: [...prev.messages, userMessage],
    }))

    try {
      const result = await sendToBackend(payload)

      if (result.thread_id) {
        setChatState((prev) => ({ ...prev, threadId: result.thread_id }))
      }

      if (result.status === 'clarification_needed') {
        setChatState((prev) => ({
          ...prev,
          mode: 'awaiting_clarification',
          messages: [
            ...prev.messages,
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              kind: 'clarification',
              content: result.question_to_user,
            },
          ],
        }))
        return
      }

      const answer = result.final_answer || {}
      const structured = result.structured_output || answer

      setChatState((prev) => ({
        ...prev,
        mode: 'ready',
        messages: [
          ...prev.messages,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            kind: 'answer',
            answer,
            structured,
          },
        ],
      }))
    } catch {
      setChatState((prev) => ({
        ...prev,
        mode: 'ready',
        messages: [
          ...prev.messages,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            kind: 'error',
            content: 'I could not answer that right now. Please try again in a moment.',
          },
        ],
      }))
    } finally {
      setDraft('')
      setIsLoading(false)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    const text = draft.trim()
    if (!text) return
    await handleUserQuestion(text)
  }

  const renderAnswer = (answer = {}) => {
    const recommendation = answer.recommendation || ''
    const evidence = answer.evidence || ''
    const citations = answer.citations || []
    const confidence = answer.confidence || 'insufficient'

    return (
      <div className="answer-card">
        {recommendation && (
          <>
            <h3>Recommendation</h3>
            <p>{recommendation}</p>
          </>
        )}

        {evidence && (
          <>
            <h3>Why this information?</h3>
            <p>{evidence}</p>
          </>
        )}

        {citations.length > 0 && (
          <>
            <div className="answer-section-heading">
              <h3>Trusted sources</h3>
              <span>Reviewed for families</span>
            </div>
            <ul className="citation-list">
              {citations.map((citation, index) => {
                const section = citation.section ? ` — ${citation.section}` : ''
                const page = citation.page !== undefined ? ` — p. ${citation.page}` : ''
                return (
                  <li key={`${citation.document}-${index}`}>
                    {citation.document}{section}{page}
                  </li>
                )
              })}
            </ul>
          </>
        )}

        <div className="confidence-line">
          <span className={`confidence-dot ${confidence}`} />
          <strong>{confidenceMap[confidence] || 'Confidence unavailable'}</strong>
          <span>Based on current guidance</span>
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <div className="ambient-shape ambient-shape-one" />
      <div className="ambient-shape ambient-shape-two" />
      <div className="app-frame">
        <header className="topbar">
          <div className="brand-lockup">
            <div className="brand-mark" aria-hidden="true"><span /></div>
            <div>
              <p className="eyebrow">Care, made clearer</p>
              <h1>Littlewell <span>health assistant</span></h1>
            </div>
          </div>
          <div className="privacy-note"><span className="status-dot" /> Private and secure</div>
        </header>

        <main className="chat-layout">
          <aside className="care-panel">
            <div className="care-illustration">
              <img src={heroImage} alt="" />
              <span className="spark spark-one">+</span>
              <span className="spark spark-two">+</span>
            </div>
            <p className="eyebrow">Your care companion</p>
            <h2>Small questions deserve thoughtful answers.</h2>
            <p className="care-copy">Get clear, practical information about your child's health, whenever you need it.</p>

            <div className="care-divider" />
            <p className="prompt-label">Try asking</p>
            <div className="suggestion-list">
              {[
                'What are signs of dehydration in children?',
                'How can I help a cough in a 2-year-old at home?',
                'How much antibiotic should I give my 3-year-old child?',
              ].map((prompt) => (
                <button className="suggestion" type="button" key={prompt} onClick={() => handleUserQuestion(prompt)} disabled={isLoading}>
                  <span>{prompt}</span><span aria-hidden="true">&#8599;</span>
                </button>
              ))}
            </div>
          </aside>
          <section className="chat-panel">
            <div className="chat-heading">
              <div className="assistant-avatar" aria-hidden="true">✦</div>
              <div>
                <h2>Ask Littlewell</h2>
                <p><span className="status-dot" /> Here to help with child health questions</p>
              </div>
            </div>
            <div className="messages">
              {chatState.messages.map((message) => {
                if (message.role === 'assistant' && message.kind === 'welcome') {
                  return (
                    <div key={message.id} className="message assistant welcome">
                      <p>{message.content}</p>
                    </div>
                  )
                }

                if (message.role === 'assistant' && message.kind === 'clarification') {
                  return (
                    <div key={message.id} className="message assistant">
                      <p>{message.content}</p>
                    </div>
                  )
                }

                if (message.role === 'assistant' && message.kind === 'error') {
                  return (
                    <div key={message.id} className="message assistant error">
                      <p>{message.content}</p>
                    </div>
                  )
                }

                if (message.role === 'assistant' && message.kind === 'answer') {
                  return (
                    <div key={message.id} className="message assistant">
                      {renderAnswer(message.answer)}
                    </div>
                  )
                }

                if (message.role === 'user') {
                  return (
                    <div key={message.id} className="message user">
                      <p>{message.content}</p>
                    </div>
                  )
                }

                return null
              })}

              {isLoading && (
                <div className="message assistant loading">
                  <p className="typing"><span /><span /><span /> Preparing a thoughtful answer</p>
                </div>
              )}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <input
                type="text"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder={
                  chatState.mode === 'awaiting_clarification'
                    ? 'Reply to the clarification question'
                    : "Ask about your child's health"
                }
                aria-label="Ask a health question"
              />
              <button type="submit" disabled={isLoading || !draft.trim()}>
                <span>Send message</span><span aria-hidden="true">&#8599;</span>
              </button>
            </form>
          </section>
        </main>
      </div>
    </div>
  )
}

export default App

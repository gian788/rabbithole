import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import { ChatInput } from './ChatInput'

function setup(props: Partial<Parameters<typeof ChatInput>[0]> = {}) {
  const onSend = vi.fn()
  const onStop = vi.fn()
  render(
    <ChatInput
      onSend={onSend}
      onStop={onStop}
      isLoading={false}
      placeholder="Ask anything..."
      {...props}
    />
  )
  return { onSend, onStop }
}

describe('ChatInput', () => {
  it('renders the placeholder text', () => {
    setup({ placeholder: 'Type here…' })
    expect(screen.getByPlaceholderText('Type here…')).toBeInTheDocument()
  })

  it('send button is disabled when input is empty', () => {
    setup()
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
  })

  it('send button is enabled after typing', async () => {
    setup()
    await userEvent.type(screen.getByRole('textbox'), 'hello')
    expect(screen.getByRole('button', { name: /send/i })).toBeEnabled()
  })

  it('calls onSend with the trimmed query on button click', async () => {
    const { onSend } = setup()
    await userEvent.type(screen.getByRole('textbox'), '  hello  ')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(onSend).toHaveBeenCalledWith('hello')
  })

  it('clears the input after sending', async () => {
    setup()
    const input = screen.getByRole('textbox')
    await userEvent.type(input, 'hello')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(input).toHaveValue('')
  })

  it('submits on Enter key', async () => {
    const { onSend } = setup()
    await userEvent.type(screen.getByRole('textbox'), 'hello{Enter}')
    expect(onSend).toHaveBeenCalledWith('hello')
  })

  it('does not submit on Shift+Enter', async () => {
    const { onSend } = setup()
    await userEvent.type(screen.getByRole('textbox'), 'hello{Shift>}{Enter}{/Shift}')
    expect(onSend).not.toHaveBeenCalled()
  })

  it('shows the stop button when loading', () => {
    setup({ isLoading: true })
    expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /send/i })).not.toBeInTheDocument()
  })

  it('calls onStop when the stop button is clicked', async () => {
    const { onStop } = setup({ isLoading: true })
    await userEvent.click(screen.getByRole('button', { name: /stop/i }))
    expect(onStop).toHaveBeenCalledOnce()
  })

  it('does not call onSend when loading and Enter is pressed', async () => {
    const { onSend } = setup({ isLoading: true })
    await userEvent.type(screen.getByRole('textbox'), 'hello{Enter}')
    expect(onSend).not.toHaveBeenCalled()
  })
})

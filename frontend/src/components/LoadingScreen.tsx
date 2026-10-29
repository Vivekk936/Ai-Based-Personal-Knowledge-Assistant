interface Props {
  variant: 'connecting' | 'reconnecting' | 'crashed'
  detail?: string
}

const COPY: Record<Props['variant'], { title: string; subtitle: string }> = {
  connecting: {
    title: 'Starting Ada',
    subtitle: 'Booting the local model backend — this takes a few seconds on first launch.',
  },
  reconnecting: {
    title: 'Reconnecting',
    subtitle: 'The backend restarted on its own. Ada will resume automatically once it responds.',
  },
  crashed: {
    title: "Ada's backend couldn't start",
    subtitle: 'Check that your NVIDIA API key is configured in .env and the FastAPI server is running, then restart Ada.',
  },
}

export function LoadingScreen({ variant, detail }: Props) {
  const { title, subtitle } = COPY[variant]
  const isError = variant === 'crashed'

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-5 bg-neutral-950">
      {isError ? (
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-500/10 text-xl text-red-400">
          !
        </div>
      ) : (
        <div className="relative h-12 w-12">
          <div className="absolute inset-0 rounded-full border-2 border-neutral-800" />
          <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-accent" />
        </div>
      )}

      <div className="max-w-xs text-center">
        <p className="text-sm font-medium text-neutral-200">{title}</p>
        <p className="mt-1.5 text-xs leading-relaxed text-neutral-500">{detail || subtitle}</p>
      </div>
    </div>
  )
}

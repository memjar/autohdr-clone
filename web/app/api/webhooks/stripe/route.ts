import { NextRequest, NextResponse } from 'next/server'
import Stripe from 'stripe'
import { kv } from '@vercel/kv'

// Only initialize Stripe if key is available
const stripe = process.env.STRIPE_SECRET_KEY
  ? new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: '2026-01-28.clover' })
  : null

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET || ''

// Credits per plan
const PLAN_CREDITS: Record<string, number> = {
  free: 10,
  standard: 500,
  enterprise: 5000,
}

export async function POST(request: NextRequest) {
  try {
    if (!stripe || !webhookSecret) {
      return NextResponse.json({ error: 'Stripe not configured' }, { status: 503 })
    }

    const body = await request.text()
    const signature = request.headers.get('stripe-signature')!

    let event: Stripe.Event

    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret)
    } catch (err: any) {
      console.error('Webhook signature verification failed:', err.message)
      return NextResponse.json({ error: 'Invalid signature' }, { status: 400 })
    }

    // Get event data as generic object to avoid type issues
    const data = event.data.object as Record<string, any>

    // Handle subscription events
    switch (event.type) {
      case 'checkout.session.completed': {
        const userId = data.metadata?.userId
        const plan = data.metadata?.plan || 'standard'

        if (userId) {
          // Store subscription info
          await kv.hset(`user:${userId}`, {
            stripeCustomerId: data.customer,
            subscriptionId: data.subscription,
            plan,
            credits: PLAN_CREDITS[plan] || 500,
            creditsUsed: 0,
            subscribedAt: new Date().toISOString(),
          })
          console.log(`User ${userId} subscribed to ${plan}`)
        }
        break
      }

      case 'customer.subscription.updated': {
        const userId = data.metadata?.userId

        if (userId) {
          const plan = data.metadata?.plan || 'standard'
          await kv.hset(`user:${userId}`, {
            plan,
            credits: PLAN_CREDITS[plan] || 500,
            status: data.status,
            currentPeriodEnd: data.current_period_end
              ? new Date(data.current_period_end * 1000).toISOString()
              : null,
          })
        }
        break
      }

      case 'customer.subscription.deleted': {
        const userId = data.metadata?.userId

        if (userId) {
          // Downgrade to free tier
          await kv.hset(`user:${userId}`, {
            plan: 'free',
            credits: PLAN_CREDITS.free,
            status: 'canceled',
            canceledAt: new Date().toISOString(),
          })
          console.log(`User ${userId} subscription canceled`)
        }
        break
      }

      case 'invoice.payment_succeeded': {
        const subscriptionId = data.subscription

        if (subscriptionId) {
          // Get subscription to find user
          const subscription = await stripe.subscriptions.retrieve(subscriptionId)
          const userId = subscription.metadata?.userId
          const plan = subscription.metadata?.plan || 'standard'

          if (userId) {
            // Reset credits for new billing period
            await kv.hset(`user:${userId}`, {
              credits: PLAN_CREDITS[plan] || 500,
              creditsUsed: 0,
              lastPaymentAt: new Date().toISOString(),
            })
            console.log(`User ${userId} credits reset for new billing period`)
          }
        }
        break
      }

      case 'invoice.payment_failed': {
        console.error('Payment failed for invoice:', data.id)
        // TODO: Send email notification to user
        break
      }
    }

    return NextResponse.json({ received: true })
  } catch (error: any) {
    console.error('Webhook error:', error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
